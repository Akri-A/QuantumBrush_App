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
from utils.helper import *

#jax.config.update("jax_enable_x64", True)

# %%
n_qubits = 2


key = jr.PRNGKey(0)

## Chooce Your Hamiltonian Ansatz
def build_hamiltonians(n_qubits, key = jr.PRNGKey(0)): 
    if n_qubits == 2:
    #    omega = jnp.array([1, 1.3])
    #    J = jnp.array([0.5])
    #    H0 = sum([omega[i]*qml.PauliZ(i) for i in range(n_qubits)] + [J[0]*qml.PauliZ(0) @ qml.PauliZ(1)])
    #    H1 = sum(qml.PauliX(i) for i in range(n_qubits-1))
        H0 = qml.PauliZ(0) @ qml.PauliZ(1)
        H1 = qml.PauliX(0) + qml.PauliX(1)
    #if n_qubits == 3: 
    #    omega = jnp.array([1, 1.1, 1.2])
    #    J = jnp.array([0.2, 0.13])
    #    H0 = sum(omega[i]*qml.PauliZ(i) for i in range(n_qubits))
    #    H1 = sum(J[i]*qml.PauliX(i)@qml.PauliX(i+1) for i in range(n_qubits-1))
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
        
    return H0, H1

key, Hkey = jr.split(key)
H0, H1 = build_hamiltonians(n_qubits, Hkey)

# %%

# %%

## Set your input state

key, inkey, outkey = jr.split(key, 3)
initial_state = jr.normal(inkey, shape=(2**n_qubits)).astype(complex)
target_state = jr.normal(outkey, shape=(2**n_qubits)).astype(complex)


initial_state /= jnp.linalg.norm(initial_state)
target_state /= jnp.linalg.norm(target_state)

# %%
n_epochs = 500
n_steps = 50
T = 2.0
lr = 0.01

## 
key, mlpkey = jax.random.split(key)

class ControlNN(eqx.Module):
    mlp : eqx.Module

    def __init__(
        self, in_size='scalar', out_size='scalar', depth=2, width_size=64, activation=jax.nn.tanh, key=mlpkey
    ):
        self.mlp = eqx.nn.MLP(
            in_size=in_size, out_size=out_size, depth=depth, width_size=width_size, activation=activation, key=key
        )

    @eqx.filter_jit 
    def __call__(self, x): 
        return (self.mlp(x))

class FourierControl(eqx.Module):
    """Fourier-based control ansatz: u(t) = a0 + Σ [a_m cos + b_m sin]."""
    a0: jnp.ndarray
    a: jnp.ndarray
    b: jnp.ndarray
    T: float
    A_max: float

    def __init__(self, key, M=6, T=1.0, A_max=1.0, scale=1e-2):
        k1, k2, k3 = jax.random.split(key, 3)
        self.a0 = jax.random.normal(k1, ()) * scale
        self.a  = jax.random.normal(k2, (M,)) * scale / jnp.arange(1, M+1)
        self.b  = jax.random.normal(k3, (M,)) * scale / jnp.arange(1, M+1)
        self.T = T
        self.A_max = A_max

    def __call__(self, t):
        """Evaluate control amplitude at time t ∈ [0, T]."""
        t = jnp.atleast_1d(t)
        freqs = jnp.arange(1, self.a.size + 1)
        cos_terms = jnp.sum(self.a * jnp.cos(2*jnp.pi*freqs[None,:]*t[:,None]/self.T), axis=-1)
        sin_terms = jnp.sum(self.b * jnp.sin(2*jnp.pi*freqs[None,:]*t[:,None]/self.T), axis=-1)
        u = self.a0 + cos_terms + sin_terms
        ## optional amplitude bound
        #u = self.A_max * jnp.tanh(u / self.A_max)
        return u if u.size > 1 else u[0]


class PiecewiseConstantControl(eqx.Module):
    amplitudes: jnp.ndarray  # shape (n_segments,)
    t_final: float
    n_segments: int

    def __call__(self, t: float):
        """Return the control amplitude u(t) for given time t."""
        idx = jnp.clip(
            (t / self.t_final * self.n_segments).astype(int),
            0,
            self.n_segments - 1,
        )
        return self.amplitudes[idx]

    def values(self, times: jnp.ndarray):
        """Convenience method: return u(t) for an array of times."""
        return jax.vmap(self.__call__)(times)

#model = ControlNN(
#   in_size='scalar', out_size='scalar', depth=3, width_size=128, activation=jax.nn.tanh, key=mlpkey
#)
#model = FourierControl(
#    key = mlpkey, M=10, T=T
#)

model = PiecewiseConstantControl(
    amplitudes=jnp.zeros(n_steps), 
    t_final= T, 
    n_segments=n_steps
)

# %%

optimizer = optax.adam(learning_rate=lr)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))



# %%
static = eqx.partition(model, eqx.is_array)[-1]
def hamiltonian(model, t, H0=H0, H1=H1):
    return H0.matrix() + model(t) * H1.matrix()

# Schrodinger
def schrodinger_rhs(t, psi, params):
    model = eqx.combine(params, static)
    H = hamiltonian(model, t)
    return -1j * (H @ psi)


# %%

# %%

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
    ## 
    ts = jnp.linspace(0, T, n_steps)
    energy = jax.scipy.integrate.trapezoid(jax.vmap((model))(ts)**2, ts)
    smooth = jax.scipy.integrate.trapezoid(jax.vmap(jax.grad(model))(ts)**2, ts)
    #return 1 - fidelity# + C*(energy + smooth)
    return -jnp.log(fidelity+1e-12)# +1e-5*(smooth+energy)


loss_fn(model, initial_state, target_state)

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





