# %%
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import equinox as eqx
import pennylane as qml
from functools import partial
import matplotlib.pyplot as plt
from utils.helper import *


jax.config.update("jax_enable_x64", True)

# %%
n_qubits = 2


key = jr.PRNGKey(0)

## Chooce Your Hamiltonian Ansatz
def build_hamiltonians(n_qubits, key = jr.PRNGKey(0)): 
    if n_qubits == 2:
        H0 = sum(qml.PauliZ(i) for i in range(2))
        H1 = qml.PauliX(0) @ qml.PauliX(1)
    else: 
        key,  Jkey = jr.split(key, 2)
        omega = 1.0 + 0.1 * jnp.arange(n_qubits)
        J =  jax.random.uniform(key=Jkey, shape=(n_qubits-1), minval=0.05, maxval=0.5)
        H0 = sum(omega[i]*qml.PauliZ(i) for i in range(n_qubits))
        H1 = sum(J[i]*qml.PauliX(i)@qml.PauliX(i+1) for i in range(n_qubits-1))
        
    return H0, H1

key, Hkey = jr.split(key)
H0, H1 = build_hamiltonians(n_qubits, Hkey)


# %%

## Set your input state
# initial_state = jnp.array([0.8, 0.6, 0.0, 0.0])
# target_state = jnp.array([0.0, 0.0, -0.6j, 0.8])

key, inkey, outkey = jr.split(key, 3)
initial_state = jr.normal(inkey, shape=(2**n_qubits))
target_state = jr.normal(outkey, shape=(2**n_qubits))


initial_state /= jnp.linalg.norm(initial_state)
target_state /= jnp.linalg.norm(target_state)

# %%
n_epochs = 1000
n_steps = 40
T = 1.0
lr = 0.02

## 
key, mlpkey = jax.random.split(key)

model = eqx.nn.MLP(
    in_size='scalar', out_size='scalar', depth=2, width_size=32, activation=jax.nn.tanh, key=mlpkey
)
optimizer = optax.adam(learning_rate=lr)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

# %%
dev = qml.device("default.qubit", wires=n_qubits)
@qml.qnode(dev)
def splitting_circuit(model, initial_state, T=1.0, n_steps=40, n= 1):
    """ 
    model: control NN
    initial_state: Initial Quantum State
    H0, H1: Hamiltanians
    T: final time
    n_steps: time steps
    n: trotterizaiton order
    """
    dt = T / n_steps
    qml.StatePrep(initial_state, wires=range(n_qubits))
    for k in range(n_steps):
        t_k = k * dt
        u_k = model(jnp.array(t_k))
        # Strang-splitting time step
        qml.ApproxTimeEvolution(H0, dt/2, n)
        qml.ApproxTimeEvolution(u_k * H1, dt, n)
        qml.ApproxTimeEvolution(H0, dt/2, n)
    return qml.state()


@eqx.filter_jit
def loss_fn(model, inital_state, target_state, T=1.0, n_steps=40, C= 3e-4):
    psi = splitting_circuit(model, inital_state, T, n_steps)
    fidelity = quantum_fidelity(psi, target_state)
    ## 
    ts = jnp.linspace(0, T, n_steps)
    integral = jax.scipy.integrate.trapezoid(jax.vmap(model)(ts)**2, ts)
    return 1 - fidelity + C*integral


# %%
@eqx.filter_jit
def make_step(model, opt_state, initial_state, target_state, T=1.0, n_steps=40):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, initial_state, target_state, T, n_steps)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


for step in range(n_epochs):
    model, opt_state, loss = make_step(model, opt_state, initial_state, target_state)
    if step % (n_epochs // 10) == 0:
        print(f"Step {step:03d}: loss = {loss:.6f}")

rho_f = splitting_circuit(model, initial_state, T, n_steps)
print("Final fidelity:", quantum_fidelity(rho_f, target_state))


# %%
def simulate_trajectory(model, initial_state, n_steps=40, T=1.0):
    dt = T / n_steps
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def step_evolution(psi_in, u_k):
        qml.StatePrep(psi_in, wires=range(n_qubits))
        qml.ApproxTimeEvolution(H0, dt/2, 1)
        qml.ApproxTimeEvolution(u_k * H1, dt, 1)
        qml.ApproxTimeEvolution(H0, dt/2, 1)
        return qml.state()

    psi = initial_state
    states = [psi]

    for k in range(n_steps):
        t_k = k * dt
        u_k = model(jnp.array(t_k))
        psi = step_evolution(psi, u_k)
        # normalize for safety
        psi = psi / jnp.linalg.norm(psi)
        states.append(psi)
    
    return jnp.stack(states)

# %%
trajectory_fidelity = jax.vmap(quantum_fidelity, in_axes=(0, None))
states = simulate_trajectory(model, initial_state, n_steps=40, T=1.0)

fidelities = trajectory_fidelity(states, target_state)
print(f"Final fidelity: {fidelities[-1]:.6f}")

# %%

times = jnp.linspace(0, 1.0, len(fidelities))
controls = jax.vmap(model)(times)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(times, fidelities)
plt.xlabel("Time")
plt.ylabel("Fidelity")
plt.title("State Fidelity over Time")

plt.subplot(1,2,2)
plt.plot(times, controls)
plt.xlabel("Time")
plt.ylabel("Control u(t)")
plt.title("Learned Control Pulse")

plt.tight_layout()
plt.show()


# %%
print(qml.draw(splitting_circuit)(model, initial_state, T=T, n_steps=n_steps))


# %%


# %%
visualize_bloch_trajectories(states, target_state, n_qubits)



