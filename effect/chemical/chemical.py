import numpy as np
import colorsys
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector, entropy, partial_trace
import importlib.util
from scipy.stats import circmean
# chemical exclusive imports
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import L_BFGS_B
from qiskit.primitives import Estimator  # local primitive

spec = importlib.util.spec_from_file_location("utils", "effect/utils.py")
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

def get_circuits_with_vqe(params):
    """Time evolution circuit for the Heisenberg model with periodic boundary conditions
    Args:
        n_qubits: number of qubits
        dt: time step
    Returns:
        circ_dt: QuantumCircuit
    """

    # --- User-controlled knobs (to integrate with the brush) ---
    bond_distance = params["user_input"]["Bond Distance"] # 0.735         # Å
    basis = params["user_input"]["Basis"] #  "sto3g"                # e.g., "sto3g", "6-31g", "cc-pVDZ", "def2-svp"
    ORDERING = params["user_input"]["Ordering"] #  "Interleaved"       # 
    print(f"bond_distance={bond_distance}, basis={basis}, ORDERING={ORDERING}")

    # --- Define molecule (H₂ with chosen bond distance / basis) ---
    driver = PySCFDriver(
        atom=f"H 0 0 0; H 0 0 {bond_distance}",
        unit=DistanceUnit.ANGSTROM,
        basis=basis
    )
    problem = driver.run()

    # --- Mapper (fermionic → qubit operator) ---
    mapper = ParityMapper(num_particles=problem.num_particles)
    qubit_op = mapper.map(problem.hamiltonian.second_q_op())

    # --- Ansatz: Hartree–Fock reference + UCCSD excitations ---
    ansatz = UCCSD(
        num_spatial_orbitals=problem.num_spatial_orbitals,
        num_particles=problem.num_particles,
        qubit_mapper=mapper,
        initial_state=HartreeFock(
            num_spatial_orbitals=problem.num_spatial_orbitals,
            num_particles=problem.num_particles,
            qubit_mapper=mapper,
        ),
    )

    # --- Optimizer & Estimator ---
    optimizer = L_BFGS_B()
    estimator = Estimator()  # Local simulation mode (your choice retained)

    # --- Iteration trace ---
    trace = {
        "metadata": {
            "bond_distance_angstrom": float(bond_distance),
            "basis": basis,
            "excitation_ordering": ORDERING
        },
        "steps": []
    }
    all_circuits = []

    def vqe_callback(eval_count, params, mean, std):
        # Ensure JSON-serializable types
        bound_circuit = ansatz.assign_parameters(params, inplace=False)
        step = {
            "iteration": int(eval_count),
            "energy_hartree": float(mean),
            "parameters": [float(x) for x in params],
            "circuit": bound_circuit,
        }
        trace["steps"].append(step)
        all_circuits.append(bound_circuit)
        # Console printout per iteration
        print(f"Iter {eval_count:02d} | E = {mean:.12f} Ha")
        print(f"Params: {step['parameters']}\n")
        
    # --- Variational Quantum Eigensolver (VQE) ---
    vqe = VQE(estimator, ansatz, optimizer, callback=vqe_callback)
    solver = GroundStateEigensolver(mapper, vqe)
    # --- Solve ground-state problem ---
    result = solver.solve(problem)
    print(len(all_circuits))

    return all_circuits


def get_chemical_colors(initial_angles,params):
    """
    Run Chemical model simulation on quantum hardware simulator.

    Args:
        dt_list: List of time steps for evolution

    Returns:
        List of RGB color tuples
    """
    # try:
    n_qubit = len(initial_angles)*2
    print("initial angles", initial_angles)

    base_circuit= get_circuits_with_vqe(params)[-1] # last_one
    circuit = QuantumCircuit(n_qubit*base_circuit.num_qubits)
    for i in range(n_qubit):
        offset = 2 * i
        qubits = [offset, offset + base_circuit.num_qubits]
        circuit.compose(base_circuit, qubits=qubits, inplace=True)
    # Note : not sure about the def of operators
    operators = [SparsePauliOp(Pauli('I'*(n_qubit-i) + p + 'I'*i)) for p in ['X','Y','Z']  for i in range(n_qubit) ]
    obs=utils.run_estimator(circuit, operators, backend=None)

    print("OBSERVABLE", type(obs), obs)
    # first qubit
    x_expectations = [ obs[k//2] for k in range(0,n_qubit,2) ]
    y_expectations = [ obs[k//2] for k in range(n_qubit,2*n_qubit,2) ]
    z_expectations = [ obs[k//2] for k in range(2*n_qubit,3*n_qubit,2) ]

    # phi = arctan2(Y, X)
    phi_expectations = [np.arctan2(y,x) % (2 * np.pi) for x, y in zip(x_expectations, y_expectations)]
    # theta = arccos(Z)
    theta_expectations = [np.arctan2(np.sqrt(x**2 + y**2),z) for x, y, z in zip(x_expectations, y_expectations, z_expectations)]

    final_angles = list(zip(phi_expectations, theta_expectations))

    print("Final angle", type(final_angles), final_angles)
    return final_angles

    
    # except Exception as e:
    #     print(f"Quantum simulation failed: {e}")

    #     return


# The main function using Chemical model
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

    path = params["stroke_input"]["path"]

    n_drops = params["user_input"]["Number of Drops"]
    assert n_drops > 0, "Number of drops must be greater than 0"

    # Split a path into n_drops smaller paths
    path_length = len(path)
    assert path_length > n_drops, "The number of pixels in the stroke must be bigger than the number of drops"

    split_size = max(1, path_length // n_drops)
    split_paths = [path[i * split_size : (i + 1) * split_size] for i in range(n_drops - 1)]
    split_paths.append(path[(n_drops - 1) * split_size :])
    # Get the radius of the drop
    radius = params["user_input"]["Radius"]
    assert radius > 0, "Radius must be greater than 0"

    initial_angles = [] #(Theta,phi)
    pixels = []
    for lines in split_paths:
        region = utils.points_within_radius(lines, radius,border = (height, width))

        selection = image[region[:, 0], region[:, 1]]
        selection = selection.astype(np.float32) / 255.0
        selection_hls = utils.rgb_to_hls(selection)
    
        phi = circmean(2 * np.pi * selection_hls[..., 0])
        theta = np.pi * np.mean(selection_hls[..., 1], axis=0)
        print("initial angle", (phi, theta))
        initial_angles.append((phi,theta))
        pixels.append((region, selection_hls))
    strength = params["user_input"]["Strength"]
    assert strength >= 0 and strength <= 1, "Strength must be between 0 and 1"

    print("Compute final angles.")
    final_angles = get_chemical_colors(initial_angles, params)
    print("final angles", final_angles)
    for i,(region,selection_hls) in enumerate(pixels):
        new_phi, new_theta = final_angles[i]
        old_phi, old_theta = initial_angles[i]

        offset_h = (new_phi - old_phi) / (2 * np.pi)
        offset_l = (new_theta - old_theta) / np.pi

        selection_hls[...,0] = (selection_hls[...,0] + offset_h) % 1
        selection_hls[...,1] += offset_l
        #selection_hls[...,2] *= 100000

        #Need to change the luminosity
        selection_hls = np.clip(selection_hls, 0, 1)

        selection_rgb = utils.hls_to_rgb(selection_hls)
        selection_rgb = (selection_rgb * 254).astype(np.uint8)

        image[region[:, 0], region[:, 1]] = selection_rgb
        
    print("Chemical effect applied successfully")
    return image
