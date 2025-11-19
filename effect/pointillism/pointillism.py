# effect/quantum_pointillism/quantum_pointillism.py
import sys
# Write to file to prove execution
with open("/app/QuantumBrush/pointillism_debug.txt", "w") as f:
    f.write("MODULE LOADED!\n")

sys.stderr.write("=" * 60 + "\n")
sys.stderr.write("POINTILLISM MODULE LOADING...\n")
sys.stderr.write("=" * 60 + "\n")
sys.stderr.flush()

try:
    import numpy as np
    import importlib.util
    import os
    sys.stderr.write("Imports successful, now loading utils...\n")
    sys.stderr.flush()

    # --- Import QuantumBrush utilities ---
    spec = importlib.util.spec_from_file_location("utils", "effect/utils.py")
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)

    sys.stderr.write("Utils loaded successfully!\n")
    sys.stderr.flush()

    # --- Import Qiskit for Quantum Circuit Implementation ---
    try:
        from qiskit import QuantumCircuit, QuantumRegister
        from qiskit.quantum_info import Statevector, SparsePauliOp
        from qiskit_aer import AerSimulator
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        QISKIT_AVAILABLE = True
        sys.stderr.write("Qiskit loaded successfully!\n")
        sys.stderr.flush()
    except ImportError as qiskit_err:
        QISKIT_AVAILABLE = False
        sys.stderr.write(f"WARNING: Qiskit not available ({qiskit_err}), will use classical fallback\n")
        sys.stderr.flush()

except Exception as e:
    sys.stderr.write(f"ERROR DURING IMPORT: {e}\n")
    sys.stderr.write(f"TRACEBACK: {str(e.__traceback__)}\n")
    sys.stderr.flush()
    raise

# Optional: Import our helper functions if you create them in separate files later
# sampling_spec = importlib.util.spec_from_file_location("utils_sampling", os.path.join(os.path.dirname(__file__), "utils_sampling.py"))
# utils_sampling = importlib.util.module_from_spec(sampling_spec)
# sampling_spec.loader.exec_module(utils_sampling)

def run(params):
    """
    Main run function for the Quantum Pointillism brush.
    Currently implements classical Poisson disk sampling for pointillism.
    The quantum Ising model logic will replace the color generation part later.
    """
    with open("/app/QuantumBrush/run_trace.txt", "w") as f:
        f.write("run() called!\n")

    import sys
    sys.stdout.flush()
    print("=" * 50, flush=True)
    print("POINTILLISM SCRIPT STARTING!", flush=True)
    print("=" * 50, flush=True)
    print("Quantum Pointillism brush started.")

    # --- 1. Extract Parameters and Inputs ---
    image = params["stroke_input"]["image_rgba"].copy().astype(np.float64) # Work with float for calculations
    path = params["stroke_input"]["path"] # Shape: (N, 2) where each row is [y, x]
    user_inputs = params["user_input"]

    dot_count = user_inputs["Dot Count"]
    target_color_hex = user_inputs["Target Color"] # e.g., "#FF5733"
    coupling_strength = user_inputs["Coupling Strength"]
    evolution_time = user_inputs["Evolution Time"]
    dot_size = user_inputs["Dot Size"]

    # Convert hex color to RGB array [0-255]
    # QuantumBrush's apply_effect.py usually handles this conversion for 'color' type,
    # so target_color should already be a numpy array like [R, G, B] (e.g., [255, 87, 51])
    target_color_rgb = target_color_hex # Already converted by apply_effect.py
    print(f"Target Color RGB: {target_color_rgb}")

    height, width = image.shape[:2]
    radius = 30  # Define a region around the stroke path for dot placement

    # Check for visualization option
    show_connections = user_inputs.get("Show Interactions", False)

    # --- 2. Get Stroke Region and Sample Dot Positions ---
    print("Sampling dot positions using improved Poisson disk...")
    region = utils.points_within_radius(path, radius, border=(height, width))
    if len(region) == 0:
        print("Warning: No region found under stroke. Returning original image.")
        return image.astype(np.uint8)

    # --- Improved Poisson Disk Sampling ---
    # Use smaller min_dist to allow more dots to fit
    min_dist = max(1.5, dot_size * 0.8)
    print(f"Using min_dist={min_dist:.1f}px for {dot_count} dots in {len(region)} region points")
    dot_positions = poisson_disk_sample_improved(region, dot_count, min_dist=min_dist)

    with open("/app/QuantumBrush/dots_sampled.txt", "w") as f:
        f.write(f"Sampled {len(dot_positions)} dots from {len(region)} region points\n")

    if len(dot_positions) == 0:
        with open("/app/QuantumBrush/error_no_dots.txt", "w") as f:
            f.write("ERROR: No dots!\n")
        return image.astype(np.uint8)

    print(f"Sampled {len(dot_positions)} dot positions.")

    # --- 3. Compute Adaptive Dot Size ---
    adjusted_dot_size = compute_dot_size_scaling(len(dot_positions), dot_count, dot_size)
    print(f"Adjusted dot size: {dot_size} -> {adjusted_dot_size}")

    # --- 4. Build Neighbor Graph ---
    neighbor_distance = compute_adaptive_neighbor_distance(dot_positions, k_neighbors=4)
    neighbors = build_neighbor_graph(dot_positions, max_distance=neighbor_distance)
    print(f"Built neighbor graph: {len(neighbors)} edges, avg distance: {neighbor_distance:.1f}px")

    # --- 5. Extract Original Colors ---
    original_colors = []
    for y, x in dot_positions:
        original_colors.append(image[y, x, :3].astype(np.uint8))

    # --- 6. Generate Colors (Quantum or Classical Fallback) ---
    print("Generating dot colors...")

    # Try quantum mode first
    quantum_circuit = create_ising_pointillism_circuit(
        N_dots=len(dot_positions),
        original_colors=original_colors,
        neighbors=neighbors,
        coupling=coupling_strength,
        evolution_time=evolution_time,
        target_color=target_color_rgb
    )

    if quantum_circuit is not None:
        # Quantum mode: Use quantum circuit
        print("  Using QUANTUM color generation")
        colors = measure_all_qubits_to_colors(quantum_circuit, len(dot_positions))

        if colors is None:
            # Measurement failed, fall back to classical
            print("  Quantum measurement failed, using classical fallback")
            colors = classical_color_blend_with_neighbors(
                dot_positions, original_colors, neighbors,
                coupling_strength, target_color_rgb
            )
    else:
        # Classical mode: Use neighbor-aware blending
        print("  Using CLASSICAL color generation (fallback)")
        colors = classical_color_blend_with_neighbors(
            dot_positions, original_colors, neighbors,
            coupling_strength, target_color_rgb
        )

    # --- 7. Draw Neighbor Connections (Optional Visualization) ---
    if show_connections and len(neighbors) > 0:
        print(f"Drawing {len(neighbors)} neighbor connections...")
        draw_neighbor_connections(image, dot_positions, neighbors,
                                 color=(128, 128, 128), alpha=64)

    # --- 8. Draw Dots ---
    print(f"Drawing {len(dot_positions)} dots...")
    for i, (y, x) in enumerate(dot_positions):
        color = colors[i]
        draw_circle(image, (y, x), adjusted_dot_size, color)

    with open("/app/QuantumBrush/complete.txt", "w") as f:
        f.write(f"Drew {len(dot_positions)} dots, returning image\n")

    print("Quantum Pointillism brush finished.")
    return image.astype(np.uint8) # Ensure output is uint8


# ==================== QUANTUM CIRCUIT PLACEHOLDERS ====================
# The following functions are placeholders for the quantum circuit implementation.
# They define the interface for quantum color evolution using the Ising model.
# TO IMPLEMENT: Replace these with actual Qiskit circuit code.

def encode_color_to_qubit(color_rgb):
    """
    Convert RGB color to quantum state angles (phi, theta) on Bloch sphere.

    This function encodes the color as a quantum state:
        |psi> = cos(theta/2)|0> + e^(i*phi) sin(theta/2)|1>

    Uses the QuantumBrush utility function to map color to spherical coordinates:
        - phi represents hue (0 to 2π)
        - theta represents lightness (0 to π)
        - saturation is stored but not used for quantum encoding

    Args:
        color_rgb: np.ndarray [R, G, B] in range [0, 255]

    Returns:
        (phi, theta): Tuple of angles in radians
    """
    phi, theta, saturation = utils.color_to_spherical(color_rgb)
    return phi, theta


def spherical_to_rgb(phi, theta, saturation=0.5):
    """
    Convert quantum state angles back to RGB color.
    This is the inverse of encode_color_to_qubit().

    Maps Bloch sphere coordinates to color:
        - phi → hue (0 to 2π → 0 to 1)
        - theta → lightness (0 to π → 0 to 1)

    Args:
        phi: Angle in [0, 2π] representing hue
        theta: Angle in [0, π] representing lightness
        saturation: Saturation value (default 0.5)

    Returns:
        np.ndarray: RGB color [R, G, B] in range [0, 255]
    """
    # Convert angles to HLS color space
    hue = phi / (2 * np.pi)
    lightness = theta / np.pi

    # Use QuantumBrush utility to convert HLS to RGB
    rgb_normalized = utils.hls_to_rgb(np.array([hue, lightness, saturation]))

    # Scale to [0, 255] range
    rgb = (rgb_normalized * 255).astype(np.uint8)
    return rgb


def create_ising_pointillism_circuit(N_dots, original_colors, neighbors,
                                      coupling, evolution_time, target_color):
    """
    Create quantum circuit for N dots with Ising model interactions.

    This implements the core quantum algorithm:
        Hamiltonian: H = -J Σ_(i,j neighbors) Z_i Z_j - h Σ_i X_i

    The circuit uses:
        1. Color-based qubit initialization (Bloch sphere encoding)
        2. RZZ gates for Ising model interactions between neighbors
        3. RX gates for external field toward target color
        4. Trotterization for time evolution

    Args:
        N_dots: Number of dots (qubits)
        original_colors: List of RGB colors for each dot
        neighbors: List of (i, j) tuples indicating which dots interact
        coupling: Coupling strength J in range [-1, 1]
                  J > 0: ferromagnetic (similar colors)
                  J < 0: antiferromagnetic (contrasting colors)
        evolution_time: Total evolution time
        target_color: RGB array for external field direction

    Returns:
        QuantumCircuit with N_dots qubits (or None if Qiskit unavailable)
    """
    # Check if Qiskit is available
    if not QISKIT_AVAILABLE:
        print("  [INFO] Qiskit not available, using classical fallback")
        return None

    # Handle edge cases
    if N_dots == 0:
        print("  [WARNING] No dots to create circuit for")
        return None

    try:
        # Create quantum circuit with N_dots qubits
        q = QuantumRegister(N_dots, 'q')
        qc = QuantumCircuit(q)

        # Step 1: Initialize each qubit with its original color
        for i, color in enumerate(original_colors):
            phi, theta = encode_color_to_qubit(color)
            # Apply rotation gates to encode color
            qc.ry(theta, i)
            qc.rz(phi, i)

        # Step 2: Time evolution using Trotterization
        # Small time steps for more accurate evolution
        dt = 0.1
        steps = max(1, int(evolution_time / dt))

        # Scale coupling strength (user input is [-1, 1], scale to appropriate range)
        J = coupling * 0.7  # Similar to notebook's coupling constant

        for step in range(steps):
            # Apply ZZ interactions (Ising model) for all neighbor pairs
            for i, j in neighbors:
                # RZZ gate implements exp(-i * theta * Z_i Z_j / 2)
                # For Ising model: theta = -2 * J * dt
                theta_ij = -2 * J * dt
                qc.rzz(theta_ij, i, j)

            # Apply per-qubit color bias (2D bias)
            # Each qubit biased toward its OWN original color
            # This implements the "2D bias" concept from teammate's code
            field_strength = 0.1

            for i in range(N_dots):
                # Bias each qubit toward its original color (not uniform target)
                phi_i, theta_i = encode_color_to_qubit(original_colors[i])
                qc.rx(field_strength * dt * theta_i, i)

        print(f"  [QUANTUM] Created circuit: {N_dots} qubits, {len(neighbors)} interactions, {steps} Trotter steps")
        return qc

    except Exception as e:
        print(f"  [ERROR] Quantum circuit creation failed: {e}")
        return None


def measure_all_qubits_to_colors(circuit, N_dots):
    """
    Measure quantum state and decode to colors.

    This function:
        1. Extracts the statevector from the quantum circuit
        2. Computes expectation values <X>, <Y>, <Z> for each qubit
        3. Reconstructs Bloch sphere angles (phi, theta) from expectation values
        4. Converts angles back to RGB colors

    The Bloch vector components give:
        <X> = sin(theta) * cos(phi)
        <Y> = sin(theta) * sin(phi)
        <Z> = cos(theta)

    Args:
        circuit: QuantumCircuit after time evolution
        N_dots: Number of qubits/dots

    Returns:
        np.ndarray of RGB colors with shape (N_dots, 3), or None if measurement fails
    """
    if not QISKIT_AVAILABLE:
        print("  [INFO] Qiskit not available for measurement")
        return None

    try:
        # Get the statevector from the circuit
        statevector = Statevector(circuit)

        colors = []

        # For each qubit, compute expectation values and decode to color
        for i in range(N_dots):
            # Create Pauli observables for this qubit
            # Format: I⊗I⊗...⊗X⊗I⊗...⊗I (X at position i)
            pauli_x = 'I' * i + 'X' + 'I' * (N_dots - i - 1)
            pauli_y = 'I' * i + 'Y' + 'I' * (N_dots - i - 1)
            pauli_z = 'I' * i + 'Z' + 'I' * (N_dots - i - 1)

            # Create SparsePauliOp observables
            obs_x = SparsePauliOp(pauli_x)
            obs_y = SparsePauliOp(pauli_y)
            obs_z = SparsePauliOp(pauli_z)

            # Compute expectation values
            exp_x = statevector.expectation_value(obs_x).real
            exp_y = statevector.expectation_value(obs_y).real
            exp_z = statevector.expectation_value(obs_z).real

            # Reconstruct angles from Bloch vector components
            # phi (hue): angle in x-y plane
            phi = np.arctan2(exp_y, exp_x) % (2 * np.pi)

            # theta (lightness): angle from z-axis
            # Clamp exp_z to [-1, 1] to avoid numerical errors in arccos
            theta = np.arccos(np.clip(exp_z, -1.0, 1.0))

            # Convert angles back to RGB color
            rgb = spherical_to_rgb(phi, theta, saturation=0.6)
            colors.append(rgb)

        print(f"  [QUANTUM] Measured {N_dots} qubits and decoded to colors")
        return np.array(colors)

    except Exception as e:
        print(f"  [ERROR] Quantum measurement failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==================== CLASSICAL COLOR FUNCTIONS ====================

def classical_color_blend_with_neighbors(dot_positions, colors_original, neighbors,
                                        coupling, target_color):
    """
    Classical fallback: Blend colors based on neighbor connectivity.
    This approximates quantum correlations using classical averaging.

    When coupling > 0 (ferromagnetic): neighbors pull toward similar colors
    When coupling < 0 (antiferromagnetic): neighbors push toward contrasting colors

    Args:
        dot_positions: np.ndarray of shape (N, 2)
        colors_original: List of RGB colors (original image colors)
        neighbors: List of (i, j) tuples
        coupling: Coupling strength in [-1, 1]
        target_color: RGB array

    Returns:
        np.ndarray of RGB colors with shape (N, 3)
    """
    N = len(dot_positions)
    colors = np.array(colors_original, dtype=np.float64)

    # Build adjacency information
    neighbor_map = {i: [] for i in range(N)}
    for i, j in neighbors:
        neighbor_map[i].append(j)
        neighbor_map[j].append(i)

    # Iterative color blending (simulate quantum evolution)
    iterations = max(1, int(10 * abs(coupling)))
    blend_rate = 0.15

    for _ in range(iterations):
        new_colors = colors.copy()

        for i in range(N):
            if len(neighbor_map[i]) > 0:
                # Average neighbor colors
                neighbor_colors = colors[neighbor_map[i]]
                avg_neighbor = np.mean(neighbor_colors, axis=0)

                if coupling > 0:
                    # Ferromagnetic: blend toward neighbors
                    new_colors[i] = (1 - blend_rate * coupling) * colors[i] + \
                                   blend_rate * coupling * avg_neighbor
                else:
                    # Antiferromagnetic: blend away from neighbors
                    contrast = 255 - avg_neighbor
                    new_colors[i] = (1 - blend_rate * abs(coupling)) * colors[i] + \
                                   blend_rate * abs(coupling) * contrast

            # Blend toward target color
            target_blend = 0.1
            new_colors[i] = (1 - target_blend) * new_colors[i] + target_blend * target_color

        colors = new_colors

    return np.clip(colors, 0, 255).astype(np.uint8)


# ==================== HELPER FUNCTIONS ====================

def poisson_disk_sample_improved(points, n_samples, min_dist=5.0, seed=None, max_attempts=30):
    """
    Improved Poisson disk sampling with greedy approach.
    More reliable than complex algorithms for small point sets.

    Args:
        points: Available pixel coordinates (from stroke region)
        n_samples: Desired number of samples
        min_dist: Minimum distance between samples
        seed: Random seed for reproducibility
        max_attempts: Maximum attempts (unused, for compatibility)

    Returns:
        np.ndarray of sampled points with shape (n, 2)
    """
    if seed is not None:
        np.random.seed(seed)

    if len(points) == 0:
        return np.array([])

    if len(points) <= n_samples:
        return points

    # Shuffle points for random selection order
    shuffled_indices = np.random.permutation(len(points))
    shuffled_points = points[shuffled_indices]

    # Greedy selection: pick points that are far enough from already selected ones
    selected = [shuffled_points[0]]

    for candidate in shuffled_points[1:]:
        # Check distance to all selected points
        distances = np.linalg.norm(selected - candidate, axis=1)

        # If far enough from all selected points, add it
        if np.all(distances >= min_dist):
            selected.append(candidate)

            # Stop if we have enough samples
            if len(selected) >= n_samples:
                break

    # If we couldn't get enough samples with min_dist, relax the constraint
    if len(selected) < n_samples:
        print(f"  Warning: Only got {len(selected)} dots with min_dist={min_dist:.1f}, relaxing constraint...")
        # Try with smaller distance
        relaxed_min_dist = min_dist * 0.7
        selected = [shuffled_points[0]]

        for candidate in shuffled_points[1:]:
            distances = np.linalg.norm(selected - candidate, axis=1)
            if np.all(distances >= relaxed_min_dist):
                selected.append(candidate)
                if len(selected) >= n_samples:
                    break

    # If still not enough, just take random points
    if len(selected) < n_samples // 2:
        print(f"  Warning: Poisson sampling got only {len(selected)} dots, using random selection...")
        # Fall back to simple random selection
        n_to_select = min(n_samples, len(points))
        indices = np.random.choice(len(points), size=n_to_select, replace=False)
        return points[indices]

    return np.array(selected)


def build_neighbor_graph(dot_positions, max_distance):
    """
    Build a graph of neighboring dots for quantum interactions.

    Args:
        dot_positions: np.ndarray of shape (N, 2) with [y, x] coordinates
        max_distance: Maximum distance for dots to be considered neighbors

    Returns:
        List of tuples (i, j) representing neighbor pairs where i < j
    """
    neighbors = []
    n_dots = len(dot_positions)

    for i in range(n_dots):
        for j in range(i + 1, n_dots):
            dist = np.linalg.norm(dot_positions[i] - dot_positions[j])
            if dist <= max_distance:
                neighbors.append((i, j))

    return neighbors


def compute_adaptive_neighbor_distance(dot_positions, k_neighbors=4):
    """
    Compute neighbor distance based on k-nearest neighbors.
    Ensures each dot has approximately k neighbors on average.

    Args:
        dot_positions: np.ndarray of shape (N, 2) with [y, x] coordinates
        k_neighbors: Target number of neighbors per dot

    Returns:
        float: Recommended neighbor distance
    """
    if len(dot_positions) < 2:
        return 10.0  # Default fallback

    # For each point, find distance to k-th nearest neighbor
    k = min(k_neighbors + 1, len(dot_positions))  # +1 to exclude self
    distances = []

    for i in range(len(dot_positions)):
        dists = []
        for j in range(len(dot_positions)):
            if i != j:
                dist = np.linalg.norm(dot_positions[i] - dot_positions[j])
                dists.append(dist)
        dists.sort()
        if len(dists) >= k_neighbors:
            distances.append(dists[k_neighbors - 1])

    # Use median of k-th neighbor distances
    if distances:
        neighbor_dist = np.median(distances)
        return neighbor_dist * 1.2  # Add 20% margin
    return 10.0


def compute_dot_size_scaling(actual_dots, requested_dots, base_size):
    """
    Scale dot size based on actual vs requested dot count.
    Maintains visual coverage when fewer dots are generated.

    Args:
        actual_dots: Number of dots actually generated
        requested_dots: Number of dots requested by user
        base_size: User-specified dot size

    Returns:
        int: Adjusted dot size
    """
    if actual_dots == 0:
        return base_size

    density_ratio = requested_dots / actual_dots
    adjusted_size = int(base_size * np.sqrt(density_ratio))
    # Clamp to reasonable range
    adjusted_size = np.clip(adjusted_size, 1, base_size * 2)
    return adjusted_size


def draw_neighbor_connections(image, dot_positions, neighbors, color=(128, 128, 128), alpha=64):
    """
    Draw lines between neighboring dots to visualize quantum interactions.

    Args:
        image: RGBA image array
        dot_positions: np.ndarray of shape (N, 2) with [y, x] coordinates
        neighbors: List of (i, j) tuples representing neighbor pairs
        color: RGB tuple for line color
        alpha: Opacity of lines (0-255)
    """
    for i, j in neighbors:
        y1, x1 = dot_positions[i]
        y2, x2 = dot_positions[j]
        draw_line(image, (y1, x1), (y2, x2), color, alpha)


def draw_line(image, p1, p2, color, alpha=64):
    """
    Draw a line between two points using Bresenham's algorithm.

    Args:
        image: RGBA image array
        p1: (y, x) start point
        p2: (y, x) end point
        color: RGB tuple
        alpha: Opacity (0-255)
    """
    y1, x1 = int(p1[0]), int(p1[1])
    y2, x2 = int(p2[0]), int(p2[1])
    height, width = image.shape[:2]

    # Bresenham's line algorithm
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        if 0 <= y1 < height and 0 <= x1 < width:
            # Blend line color with existing pixel
            image[y1, x1, :3] = (image[y1, x1, :3] * (255 - alpha) +
                                 np.array(color) * alpha) / 255
            image[y1, x1, 3] = 255

        if x1 == x2 and y1 == y2:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy


def draw_circle(image, center, radius, color):
    """
    Draws a filled circle on the image.
    """
    y, x = center
    height, width = image.shape[:2]

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy*dy + dx*dx <= radius*radius:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    image[ny, nx, :3] = color # Set RGB
                    image[ny, nx, 3] = 255 # Set Alpha to fully opaque


# --- Optional: For standalone testing (uncomment if needed) ---
# if __name__ == "__main__":
#     # Example of how you might test the run function independently
#     # This is just a conceptual example, as the full params dict is complex to mock.
#     # You would typically test by running the full QuantumBrush app.
#     test_params = {
#         "stroke_input": {
#             "image_rgba": np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8),
#             "path": np.array([[50, 50], [55, 55], [60, 60]]) # Example path
#         },
#         "user_input": {
#             "Dot Count": 20,
#             "Coupling Strength": 0.5,
#             "Evolution Time": 1.0,
#             "Target Color": np.array([255, 0, 0]), # Red
#             "Dot Size": 2
#         }
#     }
#     result_image = run(test_params)
#     print("Standalone test run completed. Shape:", result_image.shape)
