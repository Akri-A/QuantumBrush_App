#Add any dependencies but don't forget to list them in the requirements if they need to be pip installed
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector,partial_trace
from qiskit.circuit.library import RXGate, RZGate,XGate,ZGate,IGate,StatePreparation
from itertools import product
from matplotlib.path import Path


def svd(matrix=None,U=None,S=None,Vt=None):
    if U is not None:
        S_matrix = np.diag(S)  # Convert singular values into a diagonal matrix
        mat = U @ S_matrix @ Vt
        return mat

    """Compute the Ordered Singular Value Decomposition (SVD) of a matrix."""
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    sorted_indices = np.argsort(S)[::-1]  # Sort singular values in descending order
    return U[:, sorted_indices], S[sorted_indices], Vt[sorted_indices, :]


def points_within_lasso(points,border = None):
    min_x = np.min(points[:,1])
    max_x = np.max(points[:,1])+1
    min_y = np.min(points[:,0])
    max_y = np.max(points[:,0])+1

    grid = list(product(np.arange(min_y,max_y), np.arange(min_x,max_x)))
    # Create path from polygon
    path = Path(points)

    # Test which points are inside the path
    mask = path.contains_points(grid)

    # Get the pixel coordinates that are inside
    result = np.array(grid)[mask]

    if border is not None:
        result = np.clip(result, [0, 0], [border[0] - 1, border[1] - 1])

    return result


def prep(s0,s1=None): #s0 is the final state and s1 is the initial state
    if s1 is None:
        s1 = 0.5 * (np.sqrt(-3 * s0**2 + 2 * s0 + 1) - s0 + 1)
        s1 = np.clip(s1,0,1)
        #print(f"s0 {s0}")
        #print(f"s1 {s1}")
    assert s0**2 + s1**2 + s0*s1 - s0 -s1 <= 10**(-10), "Coefs must satisfy the ellipse inequality"
    return StatePreparation([np.sqrt((s0 + s1) / 2), np.sqrt((1 - s0) / 2), 0, np.sqrt((1 - s1) / 2)])


def ua_cloning(intial_angles, s0=2/3):
    '''
    Asymmetric universal cloning (same as the symetric case for default values)
    :param n_steps: Number of steps to repeat the cloning
    :param ang: Angle of the qubit to be cloned
    :return:
    '''
    num_qubits = 3
    qc = QuantumCircuit(num_qubits)

    # Rotate the first qubit to encode the image
    qc.ry(intial_angles[0],0) #theta
    qc.rz(intial_angles[1],0) #phi

    PG = prep(1-s0)
    # Creating the bell states
    qc.append(PG, [1, 2])

    qc.cx(0, 2)
    qc.cx(0, 1)
    qc.cx(2, 0)
    qc.cx(1, 0)

    sv = Statevector(qc)

    x_ops = [SparsePauliOp(Pauli('I'*(num_qubits-i) + 'X' + 'I'*i)) for i in [0,2]]
    y_ops = [SparsePauliOp(Pauli('I'*(num_qubits-i) + 'Y' + 'I'*i)) for i in [0,2]]
    z_ops = [SparsePauliOp(Pauli('I'*(num_qubits-i) + 'Z' + 'I'*i)) for i in [0,2]]

    # Calculate expectation values
    x_expectations = [sv.expectation_value(op).real for op in x_ops]
    y_expectations = [sv.expectation_value(op).real for op in y_ops]
    z_expectations = [sv.expectation_value(op).real for op in z_ops]

    return list(zip(x_expectations, y_expectations, z_expectations))

# The only thing that you need to change is this function
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

    # Create the region around those points
    copy_region = points_within_lasso(path, border = (height, width))

    # Get the RGB values of the copy region
    copy_selection = image[copy_region[:, 0], copy_region[:, 1],:3]
    copy_selection = copy_selection.astype(np.float32) / 255.0

    U,S,V = svd(copy_selection)
   
    x,y,z = np.log(S)
    mean_S = [np.mean(S)] * 3
    print("original",x,y,z)
    phi = np.mod(np.arctan2(y, x), 2 * np.pi) 
    theta = np.mod(np.arctan2(np.sqrt(x**2 + y**2), z), 2 * np.pi)

    copy_coord, paste_coord = ua_cloning((theta,phi), s0=params["user_input"]["Strength"])

    copy_r = np.linalg.norm(copy_coord)
    paste_r = np.linalg.norm(paste_coord)

    if copy_r < 10**-10:
        copy_coord = mean_S
    else:
        copy_coord = copy_r * np.exp( np.array(copy_coord) * x / copy_coord[0]  ) + (1-copy_r) * np.mean(S)

    if paste_r < 10**-10:
        paste_coord = mean_S
    else:
        paste_coord = paste_r * np.exp( np.array(paste_coord) * x / paste_coord[0]  ) + (1-paste_r) * np.mean(S)
    
    print(f"final {copy_coord} {paste_coord}")
    copy_selection = svd(U=U, S=copy_coord, Vt=V)
    paste_selection = svd(U=U, S=paste_coord, Vt=V)

    image[copy_region[:, 0], copy_region[:, 1],:3] = (copy_selection * 255).astype(np.uint8)


    paste_region = points_within_lasso(path + offset, border = (height, width))
    image[paste_region[:, 0], paste_region[:, 1],:3] = (paste_selection * 255).astype(np.uint8)

    return image
