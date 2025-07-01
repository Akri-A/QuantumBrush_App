import numpy as np
import colorsys
from qiskit import QuantumCircuit, QuantumRegister, generate_preset_pass_manager
from qiskit.quantum_info import SparsePauliOp
import importlib.util

spec = importlib.util.spec_from_file_location("utils", "effect/utils.py")
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


def scale_to_range(x, in_min=1, in_max=50, out_min=2, out_max=20):
    """
    Linearly scale x from [in_min, in_max] to [out_min, out_max].
    """
    if not (in_min <= x <= in_max):
        raise ValueError(f"Input {x} is out of range [{in_min}, {in_max}]")
    return int(round(out_min + (x - in_min) * (out_max - out_min) / (in_max - in_min)))


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

    if not isinstance(dt, (int, float)):
        dt = 0.1
    if dt <= 0:
        raise ValueError("dt must be a positive number.")

    q = QuantumRegister(n_qubits)
    circ_dt = QuantumCircuit(q)

    for n in range(n_qubits):
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

def run_heisenberg_hardware(dt_list,saturation, lightness, radius):
    """
    Run Heisenberg model simulation on quantum hardware simulator.

    Args:
        dt_list: List of time steps for evolution

    Returns:
        List of RGB color tuples
    """
    try:

        nsteps = len(dt_list)
        n_qubits = scale_to_range(radius)

        J_list =[-0.5]*n_qubits
        hz_list =[0.5]*n_qubits
        hx_list =[0.5]*n_qubits

        circuits=[]
        circ = QuantumCircuit(n_qubits)
        ### start time evolution
        for step,dt in zip(range(nsteps),dt_list):
            print(f"J_list: {J_list}, hz_list: {hz_list}, hx_list: {hx_list}, dt: {dt}")
            circ_dt = time_evolution_Heisenberg(n_qubits, J_list, hz_list, hx_list, dt)
            circ = circ.compose(circ_dt)
            circuits.append(circ.copy())

        # Create the Hamiltonian
        hamiltonian = create_heisenberg_hamiltonian(n_qubits, J_list, hz_list, hx_list)
        observables = hamiltonian

        # Run the estimator
        values=utils.run_estimator(circuits, observables, backend=None)

        values=np.array([val[0] for val in values])


        # Normalize to [0, 1]
        vmin, vmax = values.min(), values.max()
        normalized = (values - vmin) / (vmax - vmin)

        # Map normalized values to HSL-based RGB
        color_results = [
            tuple(int(round(c * 255)) for c in colorsys.hls_to_rgb(float(h),lightness,  saturation))
            for h in normalized
        ]

        return color_results

    except Exception as e:
        print(f"Quantum simulation failed: {e}")

        return


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

    lightness = params["user_input"]["Lightness"]
    assert lightness >= 0 and lightness <= 1, "Lightness must be between 0 and 1"

    
    normalized_distances = [strength] * int(max(2,min(len(path)/radius/4, 10)))
    print(normalized_distances)
    # Run Heisenberg simulation to get colors
    heisenberg_colors = run_heisenberg_hardware(normalized_distances, saturation, lightness, radius)
    # print(f"Generated {len(heisenberg_colors)} Heisenberg colors")

    # Interpolate the path to get all pixels
    interpolated_path = utils.interpolate_pixels(path)

    # Apply colors along the path
    color_idx = 0
    points_per_color = max(1, len(interpolated_path) // len(heisenberg_colors)) if heisenberg_colors else len(interpolated_path)

    for i, (x, y) in enumerate(interpolated_path):
        # Determine which color to use
        color_idx = min(i // points_per_color, len(heisenberg_colors) - 1)
        color = heisenberg_colors[color_idx] if heisenberg_colors else (255, 255, 255)

        # Get the region around this point
        region = utils.points_within_radius(np.array([[x, y]]), radius)
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