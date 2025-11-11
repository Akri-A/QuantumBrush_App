#
# @Author: chih-kang-huang
# @Date: 2025-11-10 21:30:57 
# @Last Modified by:   chih-Kang-huang
# @Last Modified time: 2025-11-10 21:30:57 
#

import jax.numpy as jnp 
import pennylane as qml
import matplotlib.pyplot as plt

def density_matrix(psi):
    """Convert a state vector to a density matrix."""
    return jnp.outer(psi, jnp.conjugate(psi))

def quantum_fidelity(psi, rho):
    psi = psi/jnp.linalg.norm(psi)
    rho = rho /jnp.linalg.norm(rho)
    return jnp.abs(jnp.vdot(psi, rho))**2


### Visualization
def von_neumann_entropy(rho):
    """Von Neumann entropy in bits."""
    eigvals = jnp.real(jnp.linalg.eigvals(rho))
    eigvals = jnp.clip(eigvals, 1e-12, 1.0)
    return -jnp.sum(eigvals * jnp.log2(eigvals))

X = jnp.array(qml.matrix(qml.PauliX(0)))
Y = jnp.array(qml.matrix(qml.PauliY(0)))
Z = jnp.array(qml.matrix(qml.PauliZ(0)))
I = jnp.eye(2)

def bloch_vector(rho):
    """Compute Bloch vector for single-qubit density matrix."""
    return jnp.array([
        jnp.real(jnp.trace(rho @ X)),
        jnp.real(jnp.trace(rho @ Y)),
        jnp.real(jnp.trace(rho @ Z))
    ])

def partial_trace(psi, keep, n_qubits):
    """Partial trace over all qubits except those in 'keep' (list of indices)."""
    rho = jnp.outer(psi, jnp.conjugate(psi))
    dims = [2] * n_qubits
    rho = rho.reshape(dims + dims)

    # Trace out all qubits not in 'keep'
    for q in reversed(range(n_qubits)):
        if q not in keep:
            rho = jnp.trace(rho, axis1=q, axis2=q + n_qubits)
            n_qubits -= 1
    return rho

# ---------- Main visualization ----------

def visualize_bloch_trajectories(states, target_state, n_qubits):
    """
    states: list/array of shape (T, 2**n)
    target_state: vector of shape (2**n,)
    n_qubits: int
    """

    # Compute single-qubit trajectories
    trajs = []
    targets = []
    for q in range(n_qubits):
        traj_q = jnp.array([
            bloch_vector(partial_trace(psi, [q], n_qubits)) for psi in states
        ])
        trajs.append(traj_q)
        targets.append(bloch_vector(partial_trace(target_state, [q], n_qubits)))

    # Entanglement entropy between qubit 0 and the rest
    ent_entropy = jnp.array([
        von_neumann_entropy(partial_trace(psi, [0], n_qubits)) for psi in states
    ])

    # ---------- Visualization ----------
    fig = plt.figure(figsize=(5 * n_qubits, 5))

    # Each qubit's Bloch trajectory
    for q in range(n_qubits):
        ax = fig.add_subplot(1, n_qubits + 1, q + 1, projection='3d')
        traj = trajs[q]
        target = targets[q]
        ax.plot(traj[:,0], traj[:,1], traj[:,2], lw=2)
        ax.scatter(traj[0,0], traj[0,1], traj[0,2], color='green', label='start')
        ax.scatter(traj[-1,0], traj[-1,1], traj[-1,2], color='red', label='end')
        ax.scatter(target[0], target[1], target[2], color='blue', label='target')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(f'Qubit {q} Bloch trajectory')
        ax.legend()

    # Entanglement entropy
    ax_e = fig.add_subplot(1, n_qubits + 1, n_qubits + 1)
    ax_e.plot(jnp.linspace(0, 1.0, len(ent_entropy)), ent_entropy, color='purple', lw=2)
    ax_e.set_xlabel('Time t')
    ax_e.set_ylabel('Entanglement entropy S(t)')
    ax_e.set_title('Entanglement entropy (qubit 0 vs rest)')
    ax_e.grid(True)

    plt.tight_layout()
    plt.show()

