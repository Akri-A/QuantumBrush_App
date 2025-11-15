import numpy as np
import colorsys
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector, entropy, partial_trace
import importlib.util
from scipy.stats import circmean

# Chemical exclusive imports
from qiskit import qpy
import json

spec = importlib.util.spec_from_file_location("utils", "effect/utils.py")
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

def resize_list_repeat(values, new_length):
    """
    Resize a list by repeating or skipping elements uniformly.
    - Expands the list by repeating elements.
    - Reduces the list by evenly skipping elements.
    """
    
    if new_length <= 0:
        return []

    old_length = len(values)
    if old_length == 0:
        return []
    if old_length == 1:
        return [values[0]] * new_length
    if old_length == new_length:
        return values.copy()

    # Compute the new indices uniformly
    idx = np.linspace(0, old_length - 1, new_length)
    idx = np.round(idx).astype(int)

    return [values[i] for i in idx]


def chemistry(initial_angles : list, circuit : QuantumCircuit, params_to_apply : list):
    """
    Run Chemical model simulation on quantum hardware simulator.

    Args:
        initial_angles (list): List of initial angles (phi, theta) for each drop
        distance (str): Bond distance for the molecule.
        circuit (QuantumCircuit): The quantum circuit in parametric form.
        params_to_apply (list): List of parameters to apply to the circuit.
        

    Returns:
        Final angles after application of circuits.
    """
    num_angles = len(initial_angles)
    print("initial angles",initial_angles)
    num_qubits_subcircuit = circuit.num_qubits 
    # expected number of subcircuits = number of qubits / number of qubits per circuit (there will be some leftover qubits if not divisible)
    num_subcircuits = num_angles // num_qubits_subcircuit 
    leftover_qubits =  num_angles % num_qubits_subcircuit # number of leftover qubits
    # adjust the number of parameters of subcircuits to match the expected number of subcircuits
    adjusted_params_to_apply = resize_list_repeat(params_to_apply, num_subcircuits)
    
    num_qubits = num_angles - leftover_qubits  # total number of qubits to use (excluding leftover)
    qc = QuantumCircuit(num_qubits+1) 
    qc.x(num_qubits)
    # Prepare each qubit in the state defined by (theta, phi)
    for i, (phi, theta) in enumerate(initial_angles[leftover_qubits:]):
        qc.ry(theta, i)
        qc.rz(phi, i)


    for index_subcircuit in range(num_subcircuits):
        add_index = index_subcircuit * num_qubits_subcircuit # starting index for the subcircuit
        index_params = index_subcircuit % len(params_to_apply) # corresponding parameters index
        qc.compose(circuit.assign_parameters({
                '_t_0_' : adjusted_params_to_apply[index_params][0], 
                '_t_1_' : adjusted_params_to_apply[index_params][1], 
                '_t_2_' : adjusted_params_to_apply[index_params][2]}), qubits= range(add_index,add_index+num_qubits_subcircuit), inplace=True)

        
    ops = [SparsePauliOp(Pauli('I'*(num_qubits-i) + p + 'I'*i)) for p in ['X','Y','Z']  for i in range(num_qubits)]
    obs = utils.run_estimator(qc,ops)

    x_expectations = obs[:num_qubits]
    y_expectations = obs[num_qubits:2*num_qubits]
    z_expectations = obs[2*num_qubits:]

    # phi = arctan2(Y, X)
    phi_expectations = [np.arctan2(y,x) % (2 * np.pi) for x, y in zip(x_expectations, y_expectations)]
    # theta = arccos(Z)
    theta_expectations = [np.arctan2(np.sqrt(x**2 + y**2),z) for x, y, z in zip(x_expectations, y_expectations, z_expectations)]

    final_angles = list(zip(phi_expectations, theta_expectations))


    return initial_angles[:leftover_qubits]+final_angles



# The main function using Chemical model
def run(params):
    """
    Executes the effect pipeline based on the provided parameters.

    Args:
        parameters (dict): A dictionary containing all the relevant data.

    Returns:
        Image: the new numpy array of RGBA values or None if the effect failed
    """

    
    # Extract image parameters
    image = params["stroke_input"]["image_rgba"]
    assert image.shape[-1] == 4, "Image must be RGBA format"

    height = image.shape[0]
    width = image.shape[1]

    # Extract user-defined parameters
    molecule = params["user_input"]["Molecule"]
    assert molecule == "H2", "Currently only H2 molecule is supported"
    with open('effect/chemical/data/' + molecule.lower() + '_parameters.json', "r") as f:
        circuit_params = json.load(f)
    with open('effect/chemical/data/' + molecule.lower() + '_circuit.qpy', "rb") as f:
        circuit = qpy.load(f)[0]

    distance = params["user_input"]["Bond Distance"]
    assert 2.5 >= distance >= 0.735, "Distance must be greater than between 0.735 and 2.5 Angstroms"
    distances = [float (d) for d in circuit_params.keys()]
    distance = min(distances, key=lambda x:abs(x-distance))
    
    radius = params["user_input"]["Radius"]
    assert radius > 0, "Radius must be greater than 0"
   
    # Extract stroke parameters
    path = params["stroke_input"]["path"]
    path_length = len(path)

    # Split path to have the same number of pixels as circuits available
    params_to_apply = circuit_params[str(distance)]
    print(f"Using distance: {distance} and the number of available circuits: {len(params_to_apply)}")
    # Number of qubits equals the number of parameters time the number of qubits per circuit
    num_circuit = params["user_input"]["Number of Circuits"] 
    split_size = max(1, path_length // num_circuit)
    split_paths = [path[i * split_size : (i + 1) * split_size] for i in range(num_circuit - 1)]
    split_paths.append(path[(num_circuit - 1) * split_size :])
    
    initial_angles = [] #(Theta,phi)
    pixels = []
    for lines in split_paths:

        region = utils.points_within_radius(lines, radius, border = (height, width))

        selection = image[region[:, 0], region[:, 1]]
        selection = selection.astype(np.float32) / 255.0
        selection_hls = utils.rgb_to_hls(selection)
    
        phi = circmean(2 * np.pi * selection_hls[..., 0])
        theta = np.pi * np.mean(selection_hls[..., 1], axis=0)
    
        initial_angles.append((phi,theta))
        pixels.append((region, selection_hls))
    
    final_angles =  chemistry(initial_angles, circuit, params_to_apply)

    for i,(region,selection_hls) in enumerate(pixels):
        new_phi, new_theta = final_angles[i]
        old_phi, old_theta = initial_angles[i]

        offset_h = (new_phi - old_phi) / (2 * np.pi)
        offset_l = (new_theta - old_theta) / np.pi

        selection_hls[...,0] = (selection_hls[...,0] + offset_h) % 1
        selection_hls[...,1] += offset_l
    
        selection_hls = np.clip(selection_hls, 0, 1)
        selection_rgb = utils.hls_to_rgb(selection_hls)
        selection_rgb = (selection_rgb * 254).astype(np.uint8)

        image[region[:, 0], region[:, 1]] = selection_rgb
        
        
    return image
