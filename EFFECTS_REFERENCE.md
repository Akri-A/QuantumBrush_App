# Quantum Brush — Effects & Brushes Reference

A guide to every brush effect available in Quantum Brush, including the quantum principles behind each one, their parameters, and credits to the original authors.

> All effects are part of the [Quantum Brush](https://github.com/moth-quantum/QuantumBrush) project by [MOTH Quantum](https://github.com/moth-quantum).
> Original source code: https://github.com/moth-quantum/QuantumBrush/tree/source

---

## 1. Basic (Acrylic)

| | |
|---|---|
| **Author** | MOTH |
| **Version** | 1.0.0 |
| **Source** | [`python/acrylic/`](https://github.com/moth-quantum/QuantumBrush/tree/source) |
| **Quantum?** | No — classical brush baseline |

A simple, uniform-color painting brush that serves as the non-quantum baseline. Paints a consistent color with configurable opacity and optional feathered edges.

**Parameters:**
- **Radius** — Brush size (0–100)
- **Color** — Base color of the brush
- **Alpha** — Opacity (0.0–1.0)
- **Blur Edges** — Smooth feathered edges vs. hard edges

---

## 2. Smudge (Damping)

| | |
|---|---|
| **Author** | Joao |
| **Version** | 1.0.0 |
| **Source** | [`python/damping/`](https://github.com/moth-quantum/QuantumBrush/tree/source) |
| **Quantum Principle** | Pumping/Damping quantum channels |
| **Dependencies** | Qiskit |

Mixes colors on the canvas using quantum amplitude damping and pumping channels. Each stroke segment is encoded as a qubit state (mapping HLS color values to Bloch sphere angles), then evolved through controlled rotation gates that simulate energy exchange with an ancilla qubit. The resulting expectation values are decoded back into shifted HLS colors.

**Parameters:**
- **Radius** — Brush size (0–100)
- **Strength** — How strongly colors are mixed (0.0–1.0)
- **Invert Luminosity** — Reverses the damping direction (light-first vs. dark-first)

---

## 3. Collage (Clone)

| | |
|---|---|
| **Author** | Joao |
| **Version** | 1.0.0 |
| **Source** | [`python/clone/`](https://github.com/moth-quantum/QuantumBrush/tree/source) |
| **Quantum Principle** | Asymmetric Universal Quantum Cloning |
| **Dependencies** | Qiskit |

Copies and pastes regions of an image subject to the no-cloning theorem of quantum mechanics. The selected region's pixel data is decomposed via SVD, and the singular values are encoded as qubit angles. An asymmetric universal cloning circuit (using a state-preparation gate and CNOT cascade) produces two imperfect copies — the quality trade-off between copy and paste fidelity is controlled by the Strength parameter. At Strength = 0.66, it achieves the symmetric cloning bound.

**Parameters:**
- **Strength** — Copy vs. paste fidelity trade-off (0.0–1.0)
  - `1.0` = perfect copy, zero paste fidelity
  - `0.0` = perfect paste, zero copy fidelity
  - `0.66` = symmetric (balanced) case

---

## 4. Aquarela (QDrop)

| | |
|---|---|
| **Author** | Joao |
| **Version** | 1.0.0 |
| **Source** | [`python/qdrop/`](https://github.com/moth-quantum/QuantumBrush/tree/source) |
| **Quantum Principle** | Entanglement-driven color mixing |
| **Dependencies** | Qiskit |

A watercolor-style brush that mixes a target color with the existing canvas colors using quantum entanglement. The stroke is split into discrete drops, each encoded as qubit states. Controlled rotations entangle each drop with an ancilla qubit, steering the color towards the target. The entanglement creates non-classical correlations between drops, producing organic color variations impossible with simple blending.

**Parameters:**
- **Number of Drops** — Resolution of the effect (2–15)
- **Radius** — Brush size (0–100)
- **Strength** — Effect intensity (0.0–1.0)
- **Target Color** — Base watercolor color

---

## 5. Heisenbrush (Continuous)

| | |
|---|---|
| **Author** | Arianna |
| **Version** | 1.0.0 |
| **Source** | [`python/heisenbrush/`](https://github.com/moth-quantum/QuantumBrush/tree/source) |
| **Quantum Principle** | Heisenberg spin model time evolution |
| **Dependencies** | Qiskit, Qiskit Aer |

Simulates how quantum spins evolve in the Heisenberg model to produce smooth, evolving color shifts along a stroke. The user's selected color is converted to spherical angles on the Bloch sphere, and each path segment becomes part of a simulated chain of qubits. The Heisenberg Hamiltonian (XX + YY + ZZ interactions with external Z and X fields) drives time evolution via Trotterized circuits. Mean magnetization is measured at each time step via the Estimator primitive and mapped back to HLS color shifts.

**Parameters:**
- **Radius** — Brush size and number of qubits in the simulation (1–100)
- **Strength** — Blend between base color and quantum-generated colors (0.0–1.0)
- **Color** — Initial state of the quantum evolution

---

## 6. Heisenbrush (Discrete)

| | |
|---|---|
| **Author** | Arianna |
| **Version** | 1.0.0 |
| **Source** | [`python/heisenbrush2/`](https://github.com/moth-quantum/QuantumBrush/tree/source) |
| **Quantum Principle** | Heisenberg spin model (discrete multi-stroke variant) |
| **Dependencies** | Qiskit, Qiskit Aer |

The discrete variant of the Heisenbrush. Instead of one continuous stroke, the user draws multiple separate strokes (up to 10), and each stroke segment independently evolves through the Heisenberg model. Each segment produces its own quantum-derived color, giving the brushwork a visibly discrete, physics-inspired variation. Uses the same underlying Heisenberg time-evolution circuit as the continuous version.

**Parameters:**
- **Radius** — Brush size and qubit count (1–100)
- **Strength** — Quantum vs. base color blend (0.0–1.0)
- **Color** — Initial quantum state color

---

## 7. Game of Life (GoL)

| | |
|---|---|
| **Author** | Daniel |
| **Version** | 1.0.0 |
| **Source** | [`python/GoL/`](https://github.com/moth-quantum/QuantumBrush/tree/source) |
| **Quantum Principle** | Quantum Cellular Automata (nearest-neighbour) |
| **Dependencies** | Qiskit, Qiskit Aer |

Implements a quantum cellular automaton inspired by Conway's Game of Life. Each pixel's HSV color is encoded as a qubit state on the Bloch sphere. For each pixel, a 9-qubit circuit is constructed from its 3x3 neighbourhood, applying controlled rotations (CRX, CRY, CRZ, CX) to simulate neighbour interactions. The central qubit's reduced density matrix is extracted via partial trace, and the closest pure state determines the new color. A semi-classical Game of Life rule (SCGOL) governs the third color channel. Supports lasso mode (Radius = 0) for region-based application.

**Parameters:**
- **Iterations** — Number of cellular automata steps (1–10)
- **Mapping** — Which HSV channels map to which GoL function (0–5)
  - `0: hsv`, `1: hvs`, `2: shv`, `3: svh`, `4: vhs`, `5: vsh`
- **Radius** — Brush width; 0 = lasso mode (0–100)

---

## 8. Chemical

| | |
|---|---|
| **Authors** | Luminists — Ali, Henrique Ennes & Jui-Ting Lu |
| **Version** | 1.0.0 |
| **Source** | [`python/chemical/`](https://github.com/moth-quantum/QuantumBrush/tree/source) |
| **Quantum Principle** | Variational Quantum Eigensolver (VQE) for molecular simulation |
| **Dependencies** | Qiskit (>=2.1.0), SciPy |

Uses intermediate steps of the VQE algorithm from quantum chemistry to modify canvas colors. Pre-computed VQE circuits for H2 molecule simulation are loaded from stored QPY files. The stroke is split into segments matching the number of available circuits. Each segment's average color is encoded as qubit angles, the VQE circuit is applied, and Pauli expectation values are measured to compute new angles. The result maps the physical process of molecular ground-state energy optimization onto visual color transformations.

**Parameters:**
- **Radius** — Brush size (0–100)
- **Bond Distance** — Simulated interatomic distance in Angstroms (0.735–2.5)
- **Number of Repeats** — How many times each circuit is applied before moving to the next (1–100)

---

## 9. Quantum Pointillism

| | |
|---|---|
| **Authors** | Khrystian Koci & Benjamin Thomas |
| **Version** | 0.2.0 |
| **Source** | [`python/pointillism/`](https://github.com/moth-quantum/QuantumBrush/tree/source) |
| **Quantum Principle** | Ising model many-body interactions |
| **Dependencies** | Qiskit, SciPy |

A pointillism brush that places dots along the stroke path with quantum-correlated colors. Dots are sampled using Poisson disk sampling with exponential distance bias from the path. Each dot's color is encoded as a qubit, and neighbouring dots interact through an Ising Hamiltonian (H = -J sum Z_i Z_j - h sum X_i). Trotterized time evolution creates quantum correlations between nearby dots. Positive coupling produces similar neighbor colors (ferromagnetic); negative coupling produces contrasting neighbors (antiferromagnetic). Falls back to classical blending for >25 dots.

**Parameters:**
- **Dot Count** — Number of dots to place (10–200)
- **Coupling Strength** — Ising coupling J (-1.0 to 1.0)
- **Evolution Time** — Quantum evolution duration (0.1–10.0)
- **Target Color** — External field color
- **Dot Size** — Radius of each dot (1–20)
- **Color Variance** — Random noise level for organic variation (0.0–1.0)

---

## 10. Steerable

| | |
|---|---|
| **Authors** | Luminists — Chih-Kang Huang & Jui-Ting Lu |
| **Version** | 1.0.0 |
| **Source** | [`python/steerable/`](https://github.com/moth-quantum/QuantumBrush/tree/source) |
| **Quantum Principle** | Quantum geometric control theory |
| **Dependencies** | Qiskit, PennyLane, JAX |

Applies quantum control theory to steer one image region's colors toward another. The user selects source and target regions via lasso, and pixel data is decomposed via SVD into normalized quantum state vectors. A PennyLane quantum circuit is trained (using JAX-based optimization) to find the unitary transformation that evolves the source state toward the target state. The learned transformation is then applied to a paste region. The time parameter `t` controls how far along the transformation the system evolves, with `t > 1` producing extrapolated effects beyond the target.

**Parameters:**
- **Controls** — Number of qubits / feature dimensions (2–4)
- **timesteps** — Discrete time steps for circuit evolution (10–100)
- **t** — Evolution time parameter (0.0–1.2)
- **Source = Paste** — Apply effect in-place or define a separate paste region
- **Show source & target** — Visualize selection outlines
- **show color** — Outline color for visualization
- **show thickness** — Outline width (1–20)

---

## Credits & Acknowledgments

| Author(s) | Effects | Affiliation |
|---|---|---|
| **MOTH** | Basic (Acrylic) | [MOTH Quantum](https://github.com/moth-quantum) |
| **Joao** | Smudge, Collage, Aquarela | MOTH Quantum contributor |
| **Arianna** | Heisenbrush (continuous & discrete) | MOTH Quantum contributor |
| **Daniel** | Game of Life | MOTH Quantum contributor |
| **Ali, Henrique Ennes & Jui-Ting Lu** | Chemical | Luminists team, MOTH Quantum contributor |
| **Khrystian Koci & Benjamin Thomas** | Quantum Pointillism | MOTH Quantum contributor |
| **Chih-Kang Huang & Jui-Ting Lu** | Steerable | Luminists team, MOTH Quantum contributor |

### Project

- **Organization:** [MOTH Quantum](https://github.com/moth-quantum)
- **Repository:** https://github.com/moth-quantum/QuantumBrush
- **Original source branch:** https://github.com/moth-quantum/QuantumBrush/tree/source
- **Issue reference:** https://github.com/moth-quantum/QuantumBrush/issues/32

### Core Dependencies

| Library | Role |
|---|---|
| [Qiskit](https://github.com/Qiskit/qiskit) | Quantum circuit construction and simulation |
| [Qiskit Aer](https://github.com/Qiskit/qiskit-aer) | Statevector simulator backend |
| [PennyLane](https://github.com/PennyLaneAI/pennylane) | Quantum circuit training (Steerable effect) |
| [JAX](https://github.com/jax-ml/jax) | Differentiable computation for optimization |
| [NumPy](https://numpy.org/) | Numerical computation |
| [SciPy](https://scipy.org/) | Scientific utilities (interpolation, circular statistics) |
