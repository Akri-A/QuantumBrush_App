# %%
# # lie_test_2qubit.py
# import numpy as np
# import itertools
# 
# # -------------------------
# # Pauli matrices & helpers
# # -------------------------
# X = np.array([[0,1],[1,0]], dtype=complex)
# Y = np.array([[0,-1j],[1j,0]], dtype=complex)
# Z = np.array([[1,0],[0,-1]], dtype=complex)
# I2 = np.eye(2, dtype=complex)
# PAULIS = {'I': I2, 'X': X, 'Y': Y, 'Z': Z}
# 
# def kron_n(ops):
#     out = ops[0]
#     for A in ops[1:]:
#         out = np.kron(out, A)
#     return out
# 
# def all_pauli_strings(n=2):
#     labels = [''.join(p) for p in itertools.product(['I','X','Y','Z'], repeat=n)]
#     mats = [kron_n([PAULIS[ch] for ch in label]) for label in labels]
#     return labels, mats
# 
# # -------------------------
# # Real flattening & lin-ind test
# # -------------------------
# def as_real_vector(A):
#     """Flatten complex matrix A into real vector [Re; Im]."""
#     re = np.real(A).ravel()
#     im = np.imag(A).ravel()
#     return np.concatenate([re, im])
# 
# def add_if_independent(basis_list, v, tol=1e-8):
#     """Add real vector v to basis_list if it increases numeric rank."""
#     if len(basis_list) == 0:
#         basis_list.append(v)
#         return True
#     M = np.stack(basis_list, axis=1)
#     rank_before = np.linalg.matrix_rank(M, tol=tol)
#     M2 = np.column_stack([M, v])
#     rank_after = np.linalg.matrix_rank(M2, tol=tol)
#     if rank_after > rank_before:
#         basis_list.append(v)
#         return True
#     return False
# 
# # -------------------------
# # Lie closure routine (specialized for 2 qubits)
# # -------------------------
# def lie_closure_2q(H_list, max_iters=30, tol=1e-8, verbose=True):
#     """
#     H_list : list of Hermitian numpy arrays (2-qubit Hamiltonians)
#     Returns: basis_mats (list of complex matrices in the Lie algebra),
#              rank_real (int), iteration log (list of tuples)
#     """
#     # Generators should be skew-Hermitian: -1j * H
#     gens = [(-1j) * H for H in H_list]
# 
#     basis = []
#     basis_real = []
# 
#     # seed with generators
#     for G in gens:
#         v = as_real_vector(G)
#         if add_if_independent(basis_real, v, tol=tol):
#             basis.append(G.copy())
# 
#     log = []
#     for it in range(max_iters):
#         new_added = 0
#         cur_basis = basis.copy()  # use snapshot so we pair newly added in same iter too
#         nb = len(cur_basis)
#         for i in range(nb):
#             for j in range(i, nb):
#                 A = cur_basis[i]
#                 B = cur_basis[j]
#                 C = A @ B - B @ A  # commutator
#                 if np.linalg.norm(C) < 1e-14:
#                     continue
#                 v = as_real_vector(C)
#                 if add_if_independent(basis_real, v, tol=tol):
#                     basis.append(C.copy())
#                     new_added += 1
#         rank_now = np.linalg.matrix_rank(np.stack(basis_real, axis=1), tol=tol) if basis_real else 0
#         log.append((it+1, new_added, rank_now, len(basis)))
#         if verbose:
#             print(f"Iter {it+1:2d}: new_added={new_added:3d}, rank_real={rank_now:2d}, total_basis={len(basis):3d}")
#         if new_added == 0:
#             break
# 
#     final_rank = np.linalg.matrix_rank(np.stack(basis_real, axis=1), tol=tol) if basis_real else 0
#     return basis, final_rank, log
# 
# # -------------------------
# # Compare to Pauli basis
# # -------------------------
# def pauli_report(basis_mats, tol=1e-6):
#     labels, mats = all_pauli_strings(2)
#     # exclude identity label 'II'
#     label_mat_pairs = [(lab, M) for lab,M in zip(labels, mats) if lab != 'II']
#     real_basis = np.stack([as_real_vector(M) for M in basis_mats], axis=1) if basis_mats else np.zeros((32,0))
#     present = []
#     missing = []
#     for lab,M in label_mat_pairs:
#         v = as_real_vector((-1j) * M)  # skew-Hermitian version for consistency
#         if real_basis.shape[1] == 0:
#             in_span = False
#         else:
#             coeffs, *_ = np.linalg.lstsq(real_basis, v, rcond=None)
#             resid = np.linalg.norm(real_basis @ coeffs - v)
#             in_span = resid < tol * max(1.0, np.linalg.norm(v))
#         if in_span:
#             present.append(lab)
#         else:
#             missing.append(lab)
#     return present, missing
# 
# # -------------------------
# # Example H0, H1 (recommended)
# # -------------------------
# # You can replace H0, H1 with your own matrices to test them.
# omega1, omega2, J = 1.0, 1.3, 0.2
# H0 = omega1 * kron_n([Z, I2]) + omega2 * kron_n([I2, Z]) + J * kron_n([Z, Z])
# H1 = kron_n([X, I2]) + kron_n([I2, X])
# 
# # -------------------------
# # Run the test
# # -------------------------
# if __name__ == "__main__":
#     print("Testing Lie closure for 2 qubits")
#     basis_mats, rank_real, log = lie_closure_2q([H0, H1], max_iters=30, tol=1e-8, verbose=True)
#     print("\nFinal real rank:", rank_real, "expected (su(4)) =", 4**2 - 1)
#     present, missing = pauli_report(basis_mats, tol=1e-6)
#     print(f"\nPauli strings present: {len(present)}; missing: {len(missing)}")
#     if missing:
#         print("Missing Pauli labels:", missing)
#     else:
#         print("All Pauli strings present -> full su(4) spanned.")


# %%
## lie_closure_nqubit.py
#import numpy as np
#import itertools
#
## ---------- Pauli definitions ----------
#X = np.array([[0,1],[1,0]], dtype=complex)
#Y = np.array([[0,-1j],[1j,0]], dtype=complex)
#Z = np.array([[1,0],[0,-1]], dtype=complex)
#I2 = np.eye(2, dtype=complex)
#PAULIS = {'I': I2, 'X': X, 'Y': Y, 'Z': Z}
#
#def kron_n(ops):
#    out = ops[0]
#    for A in ops[1:]:
#        out = np.kron(out, A)
#    return out
#
#def all_pauli_strings(n):
#    labels = [''.join(p) for p in itertools.product(['I','X','Y','Z'], repeat=n)]
#    mats = [kron_n([PAULIS[ch] for ch in label]) for label in labels]
#    return labels, mats
#
## ---------- Vectorization ----------
#def as_real_vector(A):
#    return np.concatenate([np.real(A).ravel(), np.imag(A).ravel()])
#
#def add_if_independent(basis, v, tol=1e-9):
#    if len(basis) == 0:
#        basis.append(v)
#        return True
#    M = np.stack(basis, axis=1)
#    r1 = np.linalg.matrix_rank(M, tol=tol)
#    M2 = np.column_stack([M, v])
#    r2 = np.linalg.matrix_rank(M2, tol=tol)
#    if r2 > r1:
#        basis.append(v)
#        return True
#    return False
#
## ---------- Lie closure ----------
#def lie_closure(H_list, max_iter=40, tol=1e-9, verbose=True):
#    gens = [(-1j)*H for H in H_list]
#    basis = []
#    basis_real = []
#    for G in gens:
#        add_if_independent(basis_real, as_real_vector(G), tol)
#        basis.append(G)
#
#    for it in range(max_iter):
#        new_added = 0
#        cur_basis = basis.copy()
#        for i in range(len(cur_basis)):
#            for j in range(i, len(cur_basis)):
#                C = cur_basis[i] @ cur_basis[j] - cur_basis[j] @ cur_basis[i]
#                if np.linalg.norm(C) < 1e-12:
#                    continue
#                if add_if_independent(basis_real, as_real_vector(C), tol):
#                    basis.append(C)
#                    new_added += 1
#        rank_now = np.linalg.matrix_rank(np.stack(basis_real, axis=1), tol)
#        if verbose:
#            print(f"Iter {it+1:2d}: new_added={new_added:3d}, real_rank={rank_now:4d}")
#        if new_added == 0:
#            break
#    return basis, rank_now
#
## ---------- Example system ----------
#def example_Hs(n):
#    """Return example (H0, H1) for n-qubit controllable bilinear system."""
#    # frequencies, couplings chosen to avoid degeneracy
#    omegas = np.linspace(1.0, 1.0 + 0.3*(n-1), n)
#    J = 0.3
#    # Drift H0: local Z fields + nearest-neighbor ZZ coupling
#    H0 = sum(omegas[k]*kron_n([Z if i==k else I2 for i in range(n)]) for k in range(n))
#    H0 += J * sum(kron_n([Z if i in (k,k+1) else I2 for i in range(n)]) for k in range(n-1))
#    # Control H1: global X rotation + weak nearest-neighbor XY
#    H1 = sum(kron_n([X if i==k else I2 for i in range(n)]) for k in range(n))
#    H1 += 0.2 * sum(kron_n([X if i==k else Y if i==k+1 else I2 for i in range(n)]) for k in range(n-1))
#    return H0, H1
#
## ---------- Main ----------
#if __name__ == "__main__":
#    n = 3  # choose 2, 3, or 4
#    print(f"Testing controllability for n={n} qubits ...")
#
#    H0, H1 = example_Hs(n)
#    basis, rank_real = lie_closure([H0, H1], verbose=True)
#
#    target_dim = (2**n)**2 - 1
#    print(f"\nFinal real rank = {rank_real}, expected su(2^{n}) dim = {target_dim}")
#    if rank_real == target_dim:
#        print("✅ Full controllability achieved.")
#    else:
#        print("⚠️ Not full rank — missing generators. Possibly too symmetric or degenerate H0/H1.")
#

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

jax.config.update("jax_enable_x64", True)

# %%
n_qubits = 2


key = jr.PRNGKey(0)

## Chooce Your Hamiltonian Ansatz
def build_hamiltonians(n_qubits, key = jr.PRNGKey(0)): 
    if n_qubits == 2:
        omega = jnp.array([1, 1.3])
        J = jnp.array([0.5])
        H0 = sum([omega[i]*qml.PauliZ(i) for i in range(n_qubits)] + [J[0]*qml.PauliZ(0) @ qml.PauliZ(1)])
        H1 = sum(qml.PauliX(i) for i in range(n_qubits-1))
    if n_qubits == 3: 
        omega = jnp.array([1, 1.1, 1.2])
        J = jnp.array([0.2, 0.13])
        H0 = sum(omega[i]*qml.PauliZ(i) for i in range(n_qubits))
        H1 = sum(J[i]*qml.PauliX(i)@qml.PauliX(i+1) for i in range(n_qubits-1))
    if n_qubits == 4: 
        omega = jnp.array([1, 1.12, 0.9, 1.3])
        J = jnp.array([0.2, 0.15, 0.27])
        H0 = sum(omega[i]*qml.PauliZ(i) for i in range(n_qubits))
        H1 = sum(J[i]*qml.PauliX(i)@qml.PauliX(i+1) for i in range(n_qubits-1))
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
# initial_state = jnp.array([0.8, 0.6, 0.0, 0.0])
# target_state = jnp.array([0.0, 0.0, -0.6j, 0.8])

key, inkey, outkey = jr.split(key, 3)
initial_state = jr.normal(inkey, shape=(2**n_qubits))
target_state = jr.normal(outkey, shape=(2**n_qubits))


initial_state /= jnp.linalg.norm(initial_state)
target_state /= jnp.linalg.norm(target_state)

# %%
n_epochs = 5000
n_steps = 100
T = 1.0
lr = 0.1

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
def loss_fn(model, inital_state, target_state, T=1.0, n_steps=40, C= 0):
    psi = splitting_circuit(model, inital_state, T, n_steps)
    fidelity = quantum_fidelity(psi, target_state)
    ## 
    ts = jnp.linspace(0, T, n_steps)
    energy = jax.scipy.integrate.trapezoid(jax.vmap((model))(ts)**2, ts)
    smooth = jax.scipy.integrate.trapezoid(jax.vmap(jax.grad(model))(ts)**2, ts)
    #return 1 - fidelity# + C*(energy + smooth)
    return -jnp.log(fidelity+1e-12)# +1e-5*(smooth+energy)


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
# Transpiler to universal gates
from pennylane.transforms import decompose

dev = qml.device('default.qubit')
allowed_gates = {qml.RX, qml.RY, qml.RZ, qml.CNOT}

@partial(decompose, gate_set=allowed_gates)
#@qml.compile
@qml.qnode(dev)
def circuit():
    dt = 1/n_steps

    qml.StatePrep(initial_state, wires=range(n_qubits))  # |00>
    for k in range(n_steps):
        t_k = k * dt
        u_k = model(jnp.array(t_k))
        # Strang-splitting time step
        qml.ApproxTimeEvolution(H0, dt/2, 1)
        qml.ApproxTimeEvolution(u_k * H1, dt, 1)
        qml.ApproxTimeEvolution(H0, dt/2, 1)
    
    return qml.state()
print(qml.draw(circuit)())

# %%
# %%
visualize_bloch_trajectories(states, target_state, n_qubits)


