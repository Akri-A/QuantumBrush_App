import jax.numpy as jnp 
import pennylane as qml

def density_matrix(psi):
    """Convert a state vector to a density matrix."""
    return jnp.outer(psi, jnp.conjugate(psi))

def quantum_fidelity(psi, rho):
    psi = psi/jnp.linalg.norm(psi)
    rho = rho /jnp.linalg.norm(rho)
    return jnp.abs(jnp.vdot(psi, rho))**2

def von_neumann_entropy(rho):
    vals = jnp.linalg.eigvalsh(rho)
    vals = jnp.real(vals)
    vals = jnp.clip(vals, 1e-12, 1.0)
    return -jnp.sum(vals * jnp.log2(vals))




### Visualization
X = jnp.array(qml.matrix(qml.PauliX(0)))
Y = jnp.array(qml.matrix(qml.PauliY(0)))
Z = jnp.array(qml.matrix(qml.PauliZ(0)))
I = jnp.eye(2, dtype=jnp.complex64)

def bloch_vector(rho):
    return jnp.array([
        jnp.real(jnp.trace(rho @ X)),
        jnp.real(jnp.trace(rho @ Y)),
        jnp.real(jnp.trace(rho @ Z))
    ])