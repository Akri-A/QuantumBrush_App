import numpy as np
from qiskit import generate_preset_pass_manager
from qiskit.primitives import BackendEstimatorV2 as Estimator
import os
from qiskit_aer import AerSimulator
import colorsys 



def svd(matrix=None,U=None,S=None,Vt=None):
    if U is not None:
        S_matrix = np.diag(S)  # Convert singular values into a diagonal matrix
        mat = U @ S_matrix @ Vt
        return mat

    """Compute the Ordered Singular Value Decomposition (SVD) of a matrix."""
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    sorted_indices = np.argsort(S)[::-1]  # Sort singular values in descending order
    return U[:, sorted_indices], S[sorted_indices], Vt[sorted_indices, :]


def points_within_radius(points, radius, border = None):
    """
    Given a set of points and a radius, return all points within the radius.
    Args:
        points (np.ndarray): Array of shape (N, 2) where N is the number of points.
        radius (int): The radius to search within.
        border (tuple): A tuple of (height, width)
    Returns:
        np.ndarray: Array of points within the radius.
    """
    if len(points.shape) == 1:
        points = np.array([points])

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

    if border is not None:
        result = np.clip(result, [0, 0], [border[0] - 1, border[1] - 1])

    return result


def run_estimator(circuit, operators, backend=None, options = None):
    #iqm_server_url = "https://cocos.resonance.meetiqm.com/garnet:mock"  # Replace this with the correct URL
    #provider = IQMProvider(iqm_server_url)
    #backend = provider.get_backend('garnet')
    #sampler = BackendSamplerV2(backend, options={"default_shots": 1000})
    if backend is None:
        backend = AerSimulator()

    estimator = Estimator(backend=backend, options=options)

    pm = generate_preset_pass_manager(backend=backend, optimization_level=2)

    isa_circuit = pm.run(circuit)

    if isinstance(operators,list):
        isa_observable = [op.apply_layout(isa_circuit.layout) for op in operators]
    else:
        isa_observable = operators.apply_layout(isa_circuit.layout)

    job = estimator.run([(isa_circuit, isa_observable)])

    pub_result = job.result()[0]
    obs = pub_result.data.evs
    return obs

def bresenham_line(x1, y1, x2, y2):
    points = []

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    sx = 1 if x2 > x1 else -1
    sy = 1 if y2 > y1 else -1

    if dx > dy:
        err = dx / 2.0
        while x1 != x2:
            points.append([x1, y1])
            err -= dy
            if err < 0:
                y1 += sy
                err += dx
            x1 += sx
    else:
        err = dy / 2.0
        while y1 != y2:
            points.append([x1, y1])
            err -= dx
            if err < 0:
                x1 += sx
                err += dy
            y1 += sy

    points.append([x2, y2])  # Add the last point
    return points


def interpolate_pixels(pixel_list, numpy = True):
    if len(pixel_list) == 0:
        if numpy:
            return np.array([])
        return []

    interpolated_pixels = [[pixel_list[0][0], pixel_list[0][1]]]
    # Remove consecutive duplicate pixels
    last = pixel_list[0]
    for px in pixel_list[1:]:
        if np.any(px != last):
            new_px = bresenham_line(*last,*px)
            interpolated_pixels.extend(new_px[1:])
            last = px
    if numpy:
        return np.array(interpolated_pixels)
    else:
        return interpolated_pixels


def square_region(click, radius):
    horizontal = np.arange(click[1] - radius, click[1] + radius + 1,dtype=int)
    vertical = np.arange(click[0] - radius, click[0] + radius + 1,dtype=int)
    mesh_x, mesh_y = np.meshgrid(horizontal, vertical)
    points = np.stack((mesh_y.flatten(), mesh_x.flatten()), axis=-1)
    return points


def rgb_to_hls(rgba: np.ndarray):
    """
    Convert an RGB array to HLS format.
    If the input is RGBA, the alpha channel is preserved.
    Args:
        rgba (np.ndarray): Input array of shape (N, 4) or (N, 3).
    Returns:
        np.ndarray: Converted array in HLS format.
    """

    if rgba.shape[-1] == 4:
        rgb = rgba[..., :3]

        if len(rgb.shape) == 1:
            hls = colorsys.rgb_to_hls(*rgb)
            hls.append(rgba[3])

        else:
            hls = np.apply_along_axis(lambda x: colorsys.rgb_to_hls(*x), -1, rgb)
            hls = np.concatenate([hls, rgba[..., 3][..., np.newaxis]], axis=-1)
    
    else:
        rgb = rgba

        if len(rgb.shape) == 1:
            hls = colorsys.rgb_to_hls(*rgb)

        else:
            hls = np.apply_along_axis(lambda x: colorsys.rgb_to_hls(*x), -1, rgb)

    return hls
    
def hls_to_rgb(hlsa: np.ndarray):

    if hlsa.shape[-1] == 4:
        hls = hlsa[..., :3]

        if len(hls.shape) == 1:
            rgb = colorsys.hls_to_rgb(*hls)
            rgb.append(hlsa[3])

        else:
            rgb = np.apply_along_axis(lambda x: colorsys.hls_to_rgb(*x), -1, hls)
            rgb = np.concatenate([rgb, hlsa[..., 3][..., np.newaxis]], axis=-1)
    
    else:
        hls = hlsa

        if len(hls.shape) == 1:
            rgb = colorsys.hls_to_rgb(*hls)

        else:
            rgb = np.apply_along_axis(lambda x: colorsys.hls_to_rgb(*x), -1, hls)

    return rgb
