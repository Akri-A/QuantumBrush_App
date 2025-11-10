# %%
import pennylane as qml
import jax
import jax.numpy as jnp
from jax import random, grad, jit
from functools import partial

jax.config.update("jax_enable_x64", True)

# ================================================================
# 1. Hamiltonian definitions
# ================================================================
def build_H_ops(n_qubits):
    """Build standard H0 (mixer) and H1 (nearest-neighbor Ising)."""
    H0 = sum(qml.PauliX(i) for i in range(n_qubits))
    H1 = sum(qml.PauliZ(i) @ qml.PauliZ(i + 1) for i in range(n_qubits - 1))
    return H0, H1


# ================================================================
# 2. MLP control u_theta(t)
# ================================================================
def init_mlp_params(key, hidden1=32, hidden2=32):
    """Initialize weights and biases for a 2-layer MLP."""
    k1, k2, k3 = random.split(key, 3)
    params = {
        "W1": random.normal(k1, (hidden1, 1)) * 0.5,
        "b1": jnp.zeros((hidden1,)),
        "W2": random.normal(k2, (hidden2, hidden1)) * 0.5,
        "b2": jnp.zeros((hidden2,)),
        "W3": random.normal(k3, (1, hidden2)) * 0.5,
        "b3": jnp.zeros((1,)),
    }
    return params


def mlp_u(params, t):
    """Scalar control u_theta(t) from 2-layer MLP."""
    t_in = jnp.array([[t]])
    h1 = jnp.tanh(params["W1"] @ t_in + params["b1"][:, None])
    h2 = jnp.tanh(params["W2"] @ h1 + params["b2"][:, None])
    out = params["W3"] @ h2 + params["b3"][:, None]
    return out.squeeze()


vmap_mlp_u = jax.vmap(lambda t, p: mlp_u(p, t), in_axes=(0, None))


# ================================================================
# 3. Quantum time evolution (QNode)
# ================================================================
def make_qnode(n_qubits, n_steps, T):
    """Build a QNode that evolves psi0 under H(t) = H0 + u(t)H1."""

    dev = qml.device("default.qubit", wires=n_qubits)
    H0_op, H1_op = build_H_ops(n_qubits)
    dt = T / n_steps

    @qml.qnode(dev)
    def step_evolution(u_k, psi_in):
        """Single time-step evolution."""
        qml.StatePrep(psi_in, wires=range(n_qubits))
        H_t = H0_op + u_k * H1_op
        qml.ApproxTimeEvolution(H_t, dt, 1)
        return qml.state()

    def evolve(u_vals, psi0):
        """Full evolution across time steps."""
        psi = psi0
        states = [psi0]
        for k in range(len(u_vals)):
            psi = step_evolution(u_vals[k], psi)
            states.append(psi)
        return jnp.stack(states)

    def evolve_to_time(params, psi0, t):
        """Evolve psi0 up to arbitrary time t using learned u_theta(t)."""
        n_steps_t = 50
        ts = jnp.linspace(0.0, t, n_steps_t)
        u_vals = vmap_mlp_u(ts, params)
        psi = psi0
        for u_k in u_vals:
            psi = step_evolution(u_k, psi)
        return psi

    # function to visualize final circuit
    def circuit_diagram(u_vals, psi0):
        """Return the circuit diagram for the final trained evolution."""
        @qml.qnode(dev)
        def full_evolution():
            qml.StatePrep(psi0, wires=range(n_qubits))
            for u_k in u_vals:
                H_t = H0_op + u_k * H1_op
                qml.ApproxTimeEvolution(H_t, dt, 1)
            return qml.state()
        return qml.draw(full_evolution)()

    return evolve, evolve_to_time, circuit_diagram


# ================================================================
# 4. Loss and optimization
# ================================================================
def fidelity_loss_from_params(params, evolve_fn, psi0, target, times):
    """1 - fidelity loss between final evolved state and target."""
    u_vals = vmap_mlp_u(times, params)
    states = evolve_fn(u_vals, psi0)
    psi_final = states[-1]
    overlap = jnp.vdot(target, psi_final)
    fidelity = jnp.abs(overlap) ** 2
    return 1.0 - fidelity


@partial(jit, static_argnums=(1,))
def train_step(params, evolve_fn, psi0, target, times, lr=0.1):
    """Single optimization step."""
    grads = grad(fidelity_loss_from_params)(params, evolve_fn, psi0, target, times)
    params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
    loss = fidelity_loss_from_params(params, evolve_fn, psi0, target, times)
    return params, loss


# ================================================================
# 5. Example run: Quantum Neural ODE training
# ================================================================
def example_run(
    n_qubits=2,
    n_steps=25,
    T=1.0,
    hidden1=32,
    hidden2=32,
    iters=60,
    lr=0.1,
    seed=0,
):
    key = random.PRNGKey(seed)
    params = init_mlp_params(key, hidden1, hidden2)
    times = jnp.linspace(0.0, T, n_steps)

    # Create evolution functions
    evolve_fn, evolve_to_time_fn, circuit_fn = make_qnode(n_qubits, n_steps, T)

    # Initial state |+>^n
    plus = jnp.array([1.0, 1.0]) / jnp.sqrt(2.0)
    psi0 = plus
    for _ in range(n_qubits - 1):
        psi0 = jnp.kron(psi0, plus)
    psi0 = psi0.astype(jnp.complex128)

    # Target |00...0>
    dim = 2**n_qubits
    target = jnp.zeros((dim,), dtype=jnp.complex128).at[0].set(1.0)

    print("Initial loss:", float(fidelity_loss_from_params(params, evolve_fn, psi0, target, times)))

    # Training loop
    for it in range(iters):
        params, loss = train_step(params, evolve_fn, psi0, target, times, lr=lr)
        if it % max(1, iters // 10) == 0:
            print(f"Iter {it:4d} | loss = {float(loss):.6f}")

    print("\nFinal loss:", float(fidelity_loss_from_params(params, evolve_fn, psi0, target, times)))

    # ------------------------------------------------------------
    # After training: evolve with learned control
    # ------------------------------------------------------------
    u_vals = vmap_mlp_u(times, params)
    states = evolve_fn(u_vals, psi0)
    psi_final = states[-1]
    fidelity = jnp.abs(jnp.vdot(target, psi_final)) ** 2

    print("\n=== Learned control u(t) ===")
    print(u_vals)

    print("\n=== Final Results ===")
    print(f"Final ψ(T): {psi_final}")
    print(f"Fidelity with target |0...0>: {float(fidelity):.6f}")

    # ------------------------------------------------------------
    # NEW FEATURE: evaluate ψ(t) at arbitrary time
    # ------------------------------------------------------------
    t_query = 0.5 * T
    psi_half = evolve_to_time_fn(params, psi0, t_query)
    print(f"\nψ(t={t_query}) = {psi_half}")

    # ------------------------------------------------------------
    # NEW FEATURE: print final quantum circuit
    # ------------------------------------------------------------
    print("\n=== Final Quantum Circuit (trained control) ===")
    print(circuit_fn(u_vals, psi0))

    return params, times, u_vals, psi_final, fidelity, evolve_to_time_fn


# ================================================================
# 6. Run example
# ================================================================
if __name__ == "__main__":
    params_opt, times_grid, u_vals, psi_final, fidelity, evolve_to_time_fn = example_run(
        n_qubits=2, n_steps=20, T=1.0, hidden1=32, hidden2=32, iters=50, lr=0.1, seed=42
    )

    # Example: get evolved ψ(t) at arbitrary time
    t_test = 0.75
    plus = jnp.array([1.0, 1.0]) / jnp.sqrt(2.0)
    psi0 = jnp.kron(plus, plus).astype(jnp.complex128)
    psi_t = evolve_to_time_fn(params_opt, psi0, t_test)
    print(f"\nEvolved state ψ(t={t_test:.2f}):\n{psi_t}")
