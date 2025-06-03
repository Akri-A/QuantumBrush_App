#Add any dependencies but don't forget to list them in the requirements if they need to be pip installed
import numpy as np
import colorsys
from qiskit import QuantumCircuit, QuantumRegister, transpile,generate_preset_pass_manager
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime.fake_provider import FakeManilaV2  # Using smaller backend
from qiskit_ibm_runtime import EstimatorV2 as Estimator

# Initialize backend for Heisenberg functions - using smaller backend
backend = FakeManilaV2()  # 5 qubits instead of 133

def points_within_radius(points, radius):
    """
    Given a set of points and a radius, return all points within the radius.
    Args:
        points (np.ndarray): Array of shape (N, 2) where N is the number of points.
        radius (int): The radius to search within.
    Returns:
        np.ndarray: Array of points within the radius.
    """
    assert radius > 0, "Radius must be positive"
    assert isinstance(points, np.ndarray), "Points must be a numpy array"

    # Precompute offsets within the radius
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    offsets = np.stack(np.nonzero(mask), axis=-1) - radius
    # Broadcast add offsets to all points
    all_points = points[:, None, :] + offsets[None, :, :]
    # Reshape and get unique points
    result = np.unique(all_points.reshape(-1, 2), axis=0)
    return result


def create_heisenberg_hamiltonian(n_qubits: int, J_list: list, hz_list: list, hx_list: list):
    """
    Create a periodic boundary Heisenberg model Hamiltonian as a SparsePauliOp.

    Args:
        n_qubits (int): Number of qubits (spins) in the chain
        J_list (list): List of coupling constants for each nearest neighbor
        hz_list: local Z field for each qubit
        hx_list: local X field for each qubit

    Returns:
        SparsePauliOp: The Heisenberg Hamiltonian as a sparse Pauli operator
    """
    if not isinstance(n_qubits, int) or n_qubits < 2:
        n_qubits = 4

    if not isinstance(J_list, list) or not all(isinstance(j, (int, float)) for j in J_list):
        J_list = np.random.uniform(-1, 1, n_qubits-1)
    else:
        if len(J_list) != n_qubits - 1:
            raise ValueError("Length of J_list must be equal to n_qubits - 1.")

    if not isinstance(hz_list, list) or not all(isinstance(hz, (int, float)) for hz in hz_list):
        hz_list = np.random.uniform(-1, 1, n_qubits)
    else:
        if len(hz_list) != n_qubits:
            raise ValueError("Length of hz_list must be equal to n_qubits.")

    if not isinstance(hx_list, list) or not all(isinstance(hx, (int, float)) for hx in hx_list):
        hx_list = np.random.uniform(-1, 1, n_qubits)
    else:
        if len(hx_list) != n_qubits:
            raise ValueError("Length of hx_list must be equal to n_qubits.")

    pauli_strings = []
    coefficients = []

    for i in range(n_qubits - 1):
        J = J_list[i]

        for pauli in ['X', 'Y', 'Z']:
            # Create interaction between qubits i and j
            paulistr = ['I'] * n_qubits
            paulistr[i] = paulistr[i+1] = pauli
            pauli_strings.append(''.join(paulistr))
            coefficients.append(J)

    # Add periodic boundary condition
    if n_qubits > 2:
        for pauli in ['X', 'Y', 'Z']:
            paulistr = ['I'] * n_qubits
            paulistr[0] = pauli
            paulistr[n_qubits-1] = pauli
            pauli_strings.append(''.join(paulistr))
            coefficients.append(J_list[-1])

    # Add local fields X, Z
    for i in range(n_qubits):
        for pauli, hlist in zip(['X', 'Z'], [hx_list, hz_list]):
            paulistr = ['I'] * n_qubits
            paulistr[i] = pauli
            pauli_strings.append(''.join(paulistr))
            coefficients.append(hlist[i])

    # Create the SparsePauliOp
    return SparsePauliOp(pauli_strings, coefficients)

def time_evolution_Heisenberg(n_qubits: int, J_list: list, hz_list: list, hx_list: list, dt: float) -> QuantumCircuit:
    """Time evolution circuit for the Heisenberg model with periodic boundary conditions
    Args:
        n_qubits: number of qubits
        J_list: interacting couplings for each nearest neighbor
        hz_list: local Z field for each qubit
        hx_list: local X field for each qubit
        dt: time step
    Returns:
        circ_dt: QuantumCircuit of the time evolution
    """
    if not isinstance(n_qubits, int) or n_qubits < 2:
        n_qubits = 4

    if not isinstance(J_list, list) or not all(isinstance(j, (int, float)) for j in J_list):
        J_list = np.random.uniform(-1, 1, n_qubits-1)
    else:
        if len(J_list) != n_qubits - 1:
            raise ValueError("Length of J_list must be equal to n_qubits - 1.")

    if not isinstance(hz_list, list) or not all(isinstance(hz, (int, float)) for hz in hz_list):
        hz_list = np.random.uniform(-1, 1, n_qubits)
    else:
        if len(hz_list) != n_qubits:
            raise ValueError("Length of hz_list must be equal to n_qubits.")

    if not isinstance(hx_list, list) or not all(isinstance(hx, (int, float)) for hx in hx_list):
        hx_list = np.random.uniform(-1, 1, n_qubits)
    else:
        if len(hx_list) != n_qubits:
            raise ValueError("Length of hx_list must be equal to n_qubits.")

    if not isinstance(dt, (int, float)):
        dt = 0.1
    if dt <= 0:
        raise ValueError("dt must be a positive number.")

    q = QuantumRegister(n_qubits)
    circ_dt = QuantumCircuit(q)

    for n in range(n_qubits-1):
        J = J_list[n]
        ##  exp(-it * J * (X_n X_{n+1} + Y_n Y_{n+1} + Z_n Z_{n+1}))
        circ_dt.rxx(2*J*dt, q[n], q[(n+1)%n_qubits])
        circ_dt.ryy(2*J*dt, q[n], q[(n+1)%n_qubits])
        circ_dt.rzz(2*J*dt, q[n], q[(n+1)%n_qubits])

    for n in range(n_qubits):
        ##  exp(-it * hx[n] * X_n) exp(-it * hz[n] * Z_n)
        circ_dt.rz(2*dt*hz_list[n], q[n])
        circ_dt.rx(2*dt*hx_list[n], q[n])

    return circ_dt

def run_heisenberg_hardware(dt_list,saturation, value, radius):
    """
    Run Heisenberg model simulation on quantum hardware simulator.

    Args:
        dt_list: List of time steps for evolution

    Returns:
        List of RGB color tuples
    """
    try:
        # Backend
        estimator = Estimator(backend)

        nsteps = len(dt_list)

        n_qubits = 2 if radius < 2 else min(radius, 20) #XXX: fixed to 20 qubits max

        #TODO: Hardcoding all the parameters for now
        J_list =np.random.uniform(-1, 1, n_qubits-1 )   # For Z_n Z_{n+1}
        hz_list =np.random.uniform(-1, 1, n_qubits)    # For Z_n
        hx_list =np.random.uniform(-1, 1, n_qubits)    # For X_n

        circuits=[]
        circ = QuantumCircuit(n_qubits)
        ### start time evolution
        for step,dt in zip(range(nsteps),dt_list):
            # print('step: ', step)
            circ_dt = time_evolution_Heisenberg(n_qubits, J_list, hz_list, hx_list, dt)
            circ = circ.compose(circ_dt)
            circuits.append(circ.copy())

        # Create the Hamiltonian
        hamiltonian = create_heisenberg_hamiltonian(n_qubits, J_list, hz_list, hx_list)
        observables = [hamiltonian]*len(circuits)

        # Get ISA circuits
        pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
        pubs = []
        for qc, obs in zip(circuits, observables):
            isa_circuit = pm.run(qc)
            isa_obs = obs.apply_layout(isa_circuit.layout)
            pubs.append((isa_circuit, isa_obs))

        # print('run job')
        job = estimator.run(pubs)
        job_result = job.result()
        # Extract the expectation values
        values = []
        for idx in range(len(pubs)):
            pub_result = job_result[idx]
            values.append(float(pub_result.data.evs))

        values=np.array(values)

        # Normalize to [0, 1]
        vmin, vmax = values.min(), values.max()
        normalized = (values - vmin) / (vmax - vmin)

        # Map normalized values to HSV-based RGB
        color_results = [
            tuple(int(round(c * 255)) for c in colorsys.hsv_to_rgb(float(h), saturation, value))
            for h in normalized
        ]


        return color_results

    except Exception as e:
        print(f"Quantum simulation failed: {e}")

        return


def interpolate_pixels(points):
    """
    Interpolate between points to get a continuous path of pixels.
    Args:
        points: List of (x, y) coordinate tuples
    Returns:
        List of interpolated (x, y) coordinates
    """
    if len(points) < 2:
        return points

    interpolated = []
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]

        # Calculate the distance between points
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        steps = max(dx, dy)

        if steps == 0:
            interpolated.append((x1, y1))
            continue

        # Interpolate between the two points
        for step in range(steps + 1):
            x = int(x1 + step * (x2 - x1) / steps)
            y = int(y1 + step * (y2 - y1) / steps)
            interpolated.append((x, y))

    return interpolated

# The main function using  Heisenberg model
def run(params):
    """
    Executes the Heisenberg quantum effect pipeline based on the provided parameters.

    Args:
        parameters (dict): A dictionary containing all the relevant data.

    Returns:
        Image: the new numpy array of RGBA values or None if the effect failed
    """

    # Extract image to work from
    image = params["stroke_input"]["image_rgba"].copy()
    assert image.shape[-1] == 4, "Image must be RGBA format"

    height = image.shape[0]
    width = image.shape[1]

    path = params["stroke_input"]["path"]

    # Get the radius of the effect
    radius = params["user_input"]["Radius"]
    assert radius > 0, "Radius must be greater than 0"

    strength = params["user_input"]["Strength"]
    assert strength > 0, "Strength must be greater than 0"

    saturation = params["user_input"]["Saturation"]
    assert saturation >= 0 and saturation <= 1, "Saturation must be between 0 and 1"

    value = params["user_input"]["Value"]
    assert value >= 0 and value <= 1, "Value must be between 0 and 1"

    # Calculate distances between consecutive points in the path
    distances = []
    for i in range(1, len(path)):
        prev_point = np.array(path[i-1])
        curr_point = np.array(path[i])
        dist = np.linalg.norm(curr_point - prev_point)
        distances.append(dist)

    if len(distances) == 0:
        distances = [0.1]  # Default distance if path is too short

    # Normalize distances
    max_dist = max(distances) if distances else 1.0
    normalized_distances = [ strength * d / max_dist for d in distances]

    normalized_distances=normalized_distances[:30] if len(normalized_distances) > 30 else normalized_distances #TODO: notice the fixed distances for time evolution

    # Run Heisenberg simulation to get colors
    heisenberg_colors = run_heisenberg_hardware(normalized_distances, saturation, value, radius)
    # print(f"Generated {len(heisenberg_colors)} Heisenberg colors")

    # Interpolate the path to get all pixels
    interpolated_path = interpolate_pixels(path)

    # Apply colors along the path
    color_idx = 0
    points_per_color = max(1, len(interpolated_path) // len(heisenberg_colors)) if heisenberg_colors else len(interpolated_path)

    for i, (x, y) in enumerate(interpolated_path):
        # Determine which color to use
        color_idx = min(i // points_per_color, len(heisenberg_colors) - 1)
        color = heisenberg_colors[color_idx] if heisenberg_colors else (255, 255, 255)

        # Get the region around this point
        region = points_within_radius(np.array([[x, y]]), radius)
        region = np.clip(region, [0, 0], [height - 1, width - 1])

        # Apply the color to the region
        for rx, ry in region:
            if 0 <= rx < height and 0 <= ry < width:
                # Blend the new color with the original
                original = image[rx, ry, :3].astype(np.float32)
                new_color = np.array(color, dtype=np.float32)

                # Apply blending based on strength
                blended = (1 - strength) * original + strength * new_color
                image[rx, ry, :3] = np.clip(blended, 0, 255).astype(np.uint8)

    print("Heisenberg effect applied successfully")
    return image