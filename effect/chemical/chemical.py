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



def chemistry(initial_angles : list,  distance : str, circuit : QuantumCircuit, params_to_apply : list):
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
    num_qubits = len(initial_angles)
    print("initial angles",initial_angles)

    # Prepare each qubit in the state defined by (theta, phi)
    for i, (phi, theta) in enumerate(initial_angles):
        qc = QuantumCircuit(num_qubits)
        qc.ry(theta, i)
        qc.rz(phi, i)

    # Apply the circuit with the current parameters as subcircuits
    num_qubits_circuit = circuit.num_qubits
    if num_qubits % num_qubits_circuit:
        subcircuits = list(range(0,num_qubits,num_qubits_circuit))[:-1]
    else:
        subcircuits =  list(range(0,num_qubits,num_qubits_circuit))
    for subcircuit in subcircuits:
        qc.compose(circuit.assign_parameters({
            '_t_0_' : params_to_apply[distance][i][0], 
            '_t_1_' : params_to_apply[distance][i][1], 
            '_t_2_' : params_to_apply[distance][i][2]}), qubits= subcircuit, inplace=True)
        
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


    return final_angles



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
    # Number of qubits equals the number of parameters time the number of qubits per circuit
    qubits = len(params_to_apply) * circuit.num_qubits 
    split_size = max(1, path_length // qubits)
    split_paths = [path[i * split_size : (i + 1) * split_size] for i in range(qubits - 1)]
    split_paths.append(path[(qubits - 1) * split_size :])


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


    final_angles =  chemistry(initial_angles, str(distance), circuit, params_to_apply)

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
