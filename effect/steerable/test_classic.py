# %%
# %%
# %%
#
# @Author: chih-kang-huang
# @Date: 2025-11-10 21:30:57 
# @Last Modified by:   chih-Kang-huang
# @Last Modified time: 2025-11-10 21:30:57 
#

# %%
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import equinox as eqx
import pennylane as qml
from functools import partial
import matplotlib.pyplot as plt
from utils import *
from models import *

#jax.config.update("jax_enable_x64", True)

# %%
n_qubits = 3


key = jr.PRNGKey(0)

## Chooce Your Hamiltonian Ansatz
def build_hamiltonians(n_qubits, key = jr.PRNGKey(0)): 
    H0 = sum(qml.PauliX(i) @ qml.PauliX(i+1) for i in range(n_qubits-1))   
    H0 += sum(qml.PauliY(i) @ qml.PauliY(i+1) for i in range(n_qubits-1))  
    H0 += sum(qml.PauliZ(i) @ qml.PauliZ(i+1) for i in range(n_qubits-1))

    if n_qubits == 1: 
        H_list  = [
            qml.PauliZ(0),
            qml.PauliX(0)
        ]
    elif n_qubits == 2:
        omega = jnp.array([1, 1.3])
        J = jnp.array([0.5])
        H_list = [
            qml.PauliZ(0) @ qml.PauliZ(1),
            qml.PauliY(0) @ qml.Identity(1),
            qml.PauliX(0) @ qml.Identity(1),
        ]
    elif n_qubits == 3: 
    #    omega = jnp.array([1, 1.1, 1.2])
        J = jnp.array([0.2, 0.13])
    #    H0 = sum(omega[i]*qml.PauliZ(i) for i in range(n_qubits))
    #    H1 = sum(J[i]*qml.PauliX(i)@qml.PauliX(i+1) for i in range(n_qubits-1))
        H_list  = [
            qml.PauliZ(0) @ qml.PauliZ(1) + qml.PauliZ(1) @ qml.PauliZ(2),
            qml.PauliY(0) @ qml.Identity(1) @ qml.Identity(2),
            qml.PauliX(0) @ qml.Identity(1) @ qml.Identity(2),
            sum(J[i]*qml.PauliX(i)@qml.PauliX(i+1) for i in range(n_qubits-1)),
        ]
    #if n_qubits == 4: 
    #    omega = jnp.array([1, 1.12, 0.9, 1.3])
    #    J = jnp.array([0.2, 0.15, 0.27])
    #    H0 = sum(omega[i]*qml.PauliZ(i) for i in range(n_qubits))
    #    H1 = sum(J[i]*qml.PauliX(i)@qml.PauliX(i+1) for i in range(n_qubits-1))
    #if True: 
    #    omegas = jnp.linspace(1.0, 1.0 + 0.3*(n_qubits-1), n_qubits)
    #    J = 0.3 
    #    H0 = sum(omegas[k]*qml.PauliZ(k) for k in range(n_qubits))
    #    H0 += J * sum(qml.PauliZ(k)@qml.PauliZ(k+1) for k in range(n_qubits-1))
    #    H1 = sum(qml.PauliX(k) for k in range(n_qubits))
    #    H1 += 0.2 * sum(qml.PauliX(k)@qml.PauliY(k+1) for k in range(n_qubits-1))
    else: 
        key,  Jkey = jr.split(key, 2)
        omega = 1.0 + 0.1 * jnp.arange(n_qubits)
        J =  jax.random.uniform(key=Jkey, shape=(n_qubits-1), minval=0.05, maxval=0.5)
        H0 = sum(omega[i]*qml.PauliZ(i) for i in range(n_qubits))
        H1 = sum(J[i]*qml.PauliX(i)@qml.PauliX(i+1) for i in range(n_qubits-1))
        
    return H_list

key, Hkey = jr.split(key)
H_list = build_hamiltonians(n_qubits, Hkey)



# %%

## Set your input state

key, inkey, outkey = jr.split(key, 3)
initial_state = jr.normal(inkey, shape=(2**n_qubits)).astype(complex)
target_state = jr.normal(outkey, shape=(2**n_qubits)).astype(complex)


initial_state /= jnp.linalg.norm(initial_state)
target_state /= jnp.linalg.norm(target_state)

# %%
n_epochs = 500
n_steps = 40
T = 1.0
lr = 0.05

## 
key, mlpkey = jax.random.split(key)


model = eqx.nn.MLP(
   in_size='scalar', out_size=len(H_list)-1, depth=2, width_size=16, activation=jax.nn.tanh, key=mlpkey
)

#model = PiecewiseConstantControl(
#    amplitudes=jnp.zeros((n_steps, len(H_list))), 
#    t_final= T, 
#    n_segments=n_steps
#)

# %%

optimizer = optax.adam(learning_rate=lr)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))



# %%
# %%
static = eqx.partition(model, eqx.is_array)[-1]
def hamiltonian(model, t, H_list=H_list):
    H = H_list[0].matrix()
    u = model(t)
    for i, h in enumerate(H_list[1:]): 
        H += u[i] * h.matrix()
    return H

# Schrodinger
def schrodinger_rhs(t, psi, params):
    model = eqx.combine(params, static)
    H = hamiltonian(model, t)
    return -1j * (H @ psi)



# Numerical Integrator
def propagate(psi0, t0, t1, params, steps=200):
    ts = jnp.linspace(t0, t1, steps)
    sol = jax.experimental.ode.odeint(lambda y, t, p: schrodinger_rhs(t, y, p), psi0, ts, params)
    return ts, sol

@eqx.filter_jit
def loss_fn(model, inital_state, target_state, T=1.0, n_steps=40, C= 0):
    params, static = eqx.partition(model, eqx.is_array)
    _, psi = propagate(inital_state, 0, T, params)
    fidelity = quantum_fidelity(psi[-1], target_state)
    #diff = jnp.sum( jnp.abs(psi[-1]-target_state)**2)
    #return diff
    ## 
    ts = jnp.linspace(0, T, n_steps)
    energy = jax.scipy.integrate.trapezoid(jax.vmap(lambda t : model(t)[0])(ts)**2, ts)
    energy2 = jax.scipy.integrate.trapezoid(jax.vmap(lambda t : model(t)[1])(ts)**2, ts)
    #smooth = jax.scipy.integrate.trapezoid(jax.vmap(jax.grad(model))(ts)**2, ts)
    return 1 - fidelity + 1e-5*(energy + energy2)
    #return -jnp.log(fidelity+1e-12) #+1e-5*(smooth+energy)


# %%

# %%
@eqx.filter_jit
def make_step(model, opt_state, initial_state, target_state, T=1.0, n_steps=40):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, initial_state, target_state, T, n_steps)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


for step in range(n_epochs):
    model, opt_state, loss = make_step(model, opt_state, initial_state, target_state, T=T, n_steps=n_steps)
    if step % (n_epochs // 10) == 0:
        print(f"Step {step:03d}: loss = {loss:.6f}")

params, static = eqx.partition(model, eqx.is_array)
_, rho_f = propagate(initial_state, 0, T, params)
print("Final fidelity:", quantum_fidelity(rho_f[-1], target_state))


# %%



# %%

# %%


# %%
def simulate_trajectory(model, initial_state, H_list=H_list, n_steps=40, T=1.0):
    dt = T / n_steps
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def step_evolution(psi_in, u_k):
        qml.StatePrep(psi_in, wires=range(n_qubits))
        H0 = H_list[0]
        qml.ApproxTimeEvolution(H0, dt/2, 1)
        for u, H in zip(list(u_k), H_list[1:]): 
            qml.ApproxTimeEvolution(u*H, dt/2, 1)
        for u, H in (zip(reversed(list(u_k)), reversed(H_list[1:]))): 
            qml.ApproxTimeEvolution(u*H, dt/2, 1)
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
states = simulate_trajectory(model, initial_state, n_steps=n_steps, T=T)

fidelities = trajectory_fidelity(states, target_state)
print(f"Final fidelity: {fidelities[-1]:.6f}")

# %%

times = jnp.linspace(0, T, len(fidelities))
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
# %%
visualize_bloch_trajectories(states, target_state, n_qubits)







# %%



