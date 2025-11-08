import numpy as np
# from qiskit import QuantumCircuit, generate_preset_pass_manager
# from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector,partial_trace
# from qiskit.circuit.library import RXGate, RZGate,XGate,ZGate,IGate,StatePreparation
import importlib.util
# from scipy.stats import circmean
import jax
import jax.numpy as jnp
import pennylane as qml

spec = importlib.util.spec_from_file_location("utils", "effect/utils.py")
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

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

def create_circuit_and_get_color(source_colors,target_colors,input_colors,n_qubits):
    dev = qml.device("default.qubit", wires=n_qubits)
    @qml.qnode(dev)
    def evolve_and_measure(H, t,):
        n_qb = len(H.wires)
        qml.evolve(H, t)
        obs = pauli_op_n_qubits(n_qb)
        return [qml.expval(op) for op in obs]

    # create Hamiltonian
    key = jax.random.PRNGKey(0)
    key, mlpkey = jax.random.split(key)
    params = init_mlp_params(mlpkey, [1, 16, 1])
    H = H_t(params, 1., n_qubits, steps=10)
    t = 0.5

    bloch_vec = jnp.array(evolve_and_measure(H, t))
    print("Bloch vector:", bloch_vec)
    color = (bloch_vec + 1) / 2
    print("Color:", color)
    return color

"""
Utility functions for colors
"""
def sv_to_color(sv_color):
    magnitude = np.linalg.norm(sv_color, axis=1, keepdims=True)  # shape (3,1)
    color_normalize = sv_color / magnitude 
    color_shift = (color_normalize + 1) / 2
    return magnitude, np.clip(color_shift, 0, 1)

def color_to_sv(color,magnitude):
    color_shift = 2*color-1
    return magnitude*color_shift

def create_dominant_colors(nb_controls,image,region):   
    # Get the RGB values of the copy region
    selection = image[region[:, 0], region[:, 1],:3]
    selection = selection.astype(np.float32) / 255.0

    # SVD
    U, S, Vt = np.linalg.svd(selection, full_matrices=False)
    print(f"Shapes: U: {U.shape}, S: {S.shape}, Vt: {Vt.shape}")    
    dominant_colors = (S[:, None] * Vt).T  # shape (3,3)
    magnitudes, dominant_colors = sv_to_color(dominant_colors) 
    print("Dominant_colors :)")
    for k in range(nb_controls):
        print(f"color {k+1}: {dominant_colors[k]}")
    return U, magnitudes, dominant_colors

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
    U_s, magnitudes_s, dominant_colors_s = create_dominant_colors(nb_controls,image,region_s)
    _, _, dominant_colors_t = create_dominant_colors(nb_controls,image,region_t)

    output_colors = create_circuit_and_get_color(dominant_colors_s,dominant_colors_t,dominant_colors_s,nb_controls)
    new_dominant_colors = output_colors.reshape(nb_controls, 3)
    new_dominant_colors = color_to_sv(new_dominant_colors, magnitudes_s)
    print("New dominant_colors :)")
    for k in range(nb_controls):
        print(f"color {k+1}: {new_dominant_colors[k]}")
    combined_colors = new_dominant_colors 
    if nb_controls<3:
        less_dominant_colors = (S[nb_controls:] * Vt[nb_controls:]).T  # shape (3, 3-K)
        combined_colors = np.hstack([new_dominant_colors.T, less_dominant_colors])  # shape (3,3)
    paste_selection = U_s @ combined_colors 

    paste_region = utils.points_within_lasso(path + offset, border = (height, width))
    image[paste_region[:, 0], paste_region[:, 1],:3] = (paste_selection * 255).astype(np.uint8)
    
    return image
