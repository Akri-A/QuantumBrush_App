import numpy as np
import importlib.util
import jax
import jax.numpy as jnp
import pennylane as qml

spec = importlib.util.spec_from_file_location("utils", "effect/utils.py")
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

spec = importlib.util.spec_from_file_location("steer_example", "effect/steerable/steer_example.py")
steer_example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(steer_example)

"""
Utility functions for Hamiltonians
"""
# Control term
def u(params, t):
    x = jnp.array([t])
    for i, (W, b) in enumerate(params):
        x = jnp.dot(W, x) + b
        if i < len(params) - 1:
            x = jnp.tanh(x)
    return x[0]

def standard_mixer_hamiltonian(n):
    """ 
    pennylane
    """
    return sum(qml.PauliX(i) for i in range(n))

def ising_hamiltonian(J, h):
    """
    Construct an Ising model Hamiltonian:
        H = Σ_ij J[i,j] Z_i Z_j + Σ_i h[i] Z_i
    where:
        J is a 2D array (coupling matrix)
        h is a 1D array (local fields)
    """
    n = len(h)
    H = 0

    # ZZ coupling terms
    for i in range(n):
        for j in range(i + 1, n):
            if J[i, j] != 0:
                H += J[i, j] * qml.PauliZ(i) @ qml.PauliZ(j)

    # Local fields
    for i in range(n):
        if h[i] != 0:
            H += h[i] * qml.PauliZ(i)

    return H

def H_steer(params, t, n_qubits): 
    J = jnp.zeros((n_qubits, n_qubits))
    for i in range(n_qubits - 1):
        J = J.at[i, i+1].set(1.0)  # coupling between neighbors

    h = jnp.zeros(n_qubits)  # no local field


    H_0 = standard_mixer_hamiltonian(n_qubits) 
    # H_1 = ising_matrix(J, h)
    H_1 = ising_hamiltonian(J, h)
    return H_0 + u(params, t) * H_1

def H_t(params, maxT, n_qubits, steps = 10): 
    dt = maxT/steps

    H = H_steer(params, 0, n_qubits)/2
    for k in range(1, steps):
        H += H_steer(params, k*dt, n_qubits)
    H += H_steer(params, maxT, n_qubits)/2

    
    return H * dt

"""
Utility functions for circuits
"""
# Simple MLP
def init_mlp_params(key, sizes):
    keys = jax.random.split(key, len(sizes) - 1)
    params = []
    for k, (n_in, n_out) in zip(keys, zip(sizes[:-1], sizes[1:])):
        W = jax.random.normal(k, (n_out, n_in)) * jnp.sqrt(2 / (n_in + n_out))
        b = jnp.zeros((n_out,))
        params.append((W, b))
    return params

def pauli_op_single_qubit(P, target, n_qubits):
    """Tensor product: I ⊗ ... ⊗ P ⊗ ... ⊗ I"""
    op = None
    for i in range(n_qubits):
        single = P(i) if i == target else qml.Identity(i)
        op = single if op is None else op @ single
    return op

def pauli_op_n_qubits(n_qubits):
    observable = []
    for target in range(n_qubits):
        observable.append(pauli_op_single_qubit(qml.PauliX, target, n_qubits))
        observable.append(pauli_op_single_qubit(qml.PauliY, target, n_qubits))
        observable.append(pauli_op_single_qubit(qml.PauliZ, target, n_qubits))
    return observable

"""
Utility functions for colors
"""

def pixels_to_angles(pixel_vectors):
    """
    Convert an array of RGB pixel vectors into rotation angles using SVD.

    The angles returned convert these 3-D directions into spherical angles
    (theta, phi), suitable for parameterizing qubit rotations.

    Parameters
    ----------
    pixel_vectors : array-like, shape (N, 3)
        N RGB vectors (3-tuples valued in [0, 1]).
    eps : float
        Small stabilizer to avoid log(0).

    Returns
    -------
    angles : list of float, length 8.
    """

    pixel_vectors = np.asarray(pixel_vectors)
    if pixel_vectors.ndim != 2 or pixel_vectors.shape[1] != 3:
        raise ValueError("pixel_vectors must be of shape (N, 3).")

    # SVD: Vt rows represent orthonormal color directions (principal components)
    U, S, Vt = np.linalg.svd(pixel_vectors, full_matrices=False)

    def to_spherical(v):
        """Map a 3D direction to spherical angles (theta, phi)."""
        v = v / np.linalg.norm(v)
        x, y, z = v
        phi = np.arctan2(y, x)
        theta = np.arccos(z) 
        return theta, phi

    # 1 pair from global singular values
    angles = list(to_spherical(np.log(S)))
    print(f"Dominant values (S): {S}")

    # 3 pairs from color basis directions (rows of Vt)
    print("Dominant colors (rows of Vt):")
    for color_dir in Vt:
        angles.extend(to_spherical(color_dir))
        print(color_dir)

    return angles

def selection_to_angles(image, region):
    pixels = image[region[:, 0], region[:, 1],:3]
    pixels = pixels.astype(np.float32) / 255.0
    return pixels_to_angles(pixels)

def measures_to_pixels(template, measures, nb_controls):
    """
    Reconstruct a pixel vector/matrix from measured angles via SVD.

    Parameters
    ----------
    template : array-like, shape (N, 3)
    measures : array-like, shape (12,)
        Measured values:
        - measures[:3] -> log-scaled singular values
        - measures[3:] -> flattened 3x3 matrix (rows of V^T)

    Returns
    -------
    reconstructed : ndarray, shape (N, 3)
        Reconstructed pixel vectors using the measured S and V^T.
    """
    template = np.asarray(template)
    measures = np.asarray(measures)
    
    if measures.size != 3*nb_controls:
        raise ValueError(f"measures must be of length {3*nb_controls} (3 singular values + {3*(nb_controls-1)} matrix entries).")
    
    U, S, Vt = np.linalg.svd(template, full_matrices=False)
    x,y,z = np.log(S)
    r = np.linalg.norm([x,y,z])
    mean_S = [np.mean(S)] * 3
    s_values = measures[:3]
    s_values_r = np.linalg.norm(s_values)
    if s_values_r < 10**-10:
        s_values = mean_S
    else:
        s_values = s_values_r * np.exp( np.array(s_values) * r / s_values_r  ) + (1-s_values_r) * np.mean(S)
    new_S = np.diag(s_values)
    print(f"New dominant values (S): {s_values}")

    measured_Vt = measures[3:].reshape(nb_controls - 1, 3).copy()
    new_Vt = np.zeros((3, 3))
    print("New dominant colors (rows of Vt):")
    for i in range(3):
        if i < nb_controls - 1:
            new_Vt[i] = np.linalg.norm(Vt[i]) * measured_Vt[i]
        else:
            new_Vt[i] = Vt[i]
        print(new_Vt[i])
    reconstructed = U @ new_S @ new_Vt
    return reconstructed

def YZEmbedding(angles, wires):
    angles = np.asarray(angles)
    angles = angles.reshape(len(wires), 2)
    qml.AngleEmbedding(angles[:, 0], wires=wires, rotation="Y")
    qml.AngleEmbedding(angles[:, 1], wires=wires, rotation="Z")

"""
Mesurement
"""
def create_circuit_and_measure(params, source_angles, target_angles, input_angles, n_qubits):
    #  angles: array-like, shape (2*n_qubit,) 
    #         A flattened array containing 2*n_qubit angle values, representing
    #         n pairs of (theta, phi) rotation angles.
    dev = qml.device("default.qubit", wires=n_qubits)
    T = params["user_input"]["max T"]
    n_T = params["user_input"]["timesteps"]
    dt = T/n_T
    print(f"T={T}, n_T={n_T}, dt={dt}")
     
    @qml.qnode(dev)
    def evolve_and_measure_with_input_angles(H, t, angles_input):
        n_q = len(angles_input)//2
        # Apply input embedding
        YZEmbedding(angles_input, wires=range(n_q))
        print("angles_input:", angles_input)
        print("len(angles_input)//2:", len(angles_input)//2)
    
        # Time evolution
        qml.ApproxTimeEvolution(H, t, n_T)
        
        # Measurements
        obs = pauli_op_n_qubits(n_q)
        print("Observables:", obs)
        return [qml.expval(op) for op in obs]
    
    # # create Hamiltonian
    H = None
    if params["user_input"]["Test"] :
        key = jax.random.PRNGKey(0)
        key, mlpkey = jax.random.split(key)
        params0 = init_mlp_params(mlpkey, [1, 16, 1])
        H = H_t(params0, 1., n_qubits, steps=10)
    else : # First example without learning
        H0_op, H1_op = steer_example.build_H_ops(n_qubits)
        u = 2
        H = H0_op + u * H1_op
    t = params["user_input"]["t"]*T
    print(f"t={t}")
    measures = jnp.array(evolve_and_measure_with_input_angles(H, t, source_angles))

    print("Bloch vector measures (XYZ):", measures)
    return measures

"""
Brush execution 
"""
def run(params):
    """
    Executes the effect pipeline based on the provided parameters.

    Args:
        parameters (dict): A dictionary containing all the relevant data.

    Returns:
        Image: the new numpy array of RGBA values or None if the effect failed
    """
    
    # Extract image to work from
    image = params["stroke_input"]["image_rgba"]
    # It's a good practice to check any of the request variables
    assert image.shape[-1] == 4, "Image must be RGBA format"

    height = image.shape[0]
    width = image.shape[1]

    # Extract the copy and past points
    clicks = params["stroke_input"]["clicks"]
    assert len(clicks) == 2, "The number of clicks must 2, i.e. copy and paste"

    offset = clicks[1]-clicks[0]

    # Extract the lasso path
    path = params["stroke_input"]["path"]
    
    # Remove any leftover points
    while np.all(path[-1] != clicks[-1]):
        path = path[:-1]
    path = path[:-1] #Remove the last click

    nb_controls = params["user_input"]["Controls"]

    # Create the regions (source and target)
    region_s = utils.points_within_lasso(path, border = (height, width))
    region_t = utils.points_within_lasso(path + offset, border = (height, width))

    ####### test
    print("=== Computing angles from source ===")
    angles_s = selection_to_angles(image, region_s)
    print("=== Computing angles from target ===")
    angles_t = selection_to_angles(image, region_t)
    output_measures = create_circuit_and_measure(params, angles_s[:nb_controls*2], angles_t[:nb_controls*2], angles_s[:nb_controls*2], nb_controls)

    region_in = region_s
    pixels = image[region_in[:, 0], region_in[:, 1],:3]
    pixels = pixels.astype(np.float32) / 255.0
    new_pixels = measures_to_pixels(pixels, output_measures,nb_controls)
    paste_region = utils.points_within_lasso(path + offset, border = (height, width))
    image[paste_region[:, 0], paste_region[:, 1],:3] = (new_pixels * 255).astype(np.uint8)

    return image
