#
# @Author: chih-kang-huang
# @Date: 2025-11-10 21:30:57 
# @Last Modified by:   chih-Kang-huang
# @Last Modified time: 2025-11-10 21:30:57 
#

import pennylane as qml
import jax.numpy as jnp

def splitting_circuit(model, initial_state, H0, H1, n_qubits, T=1.0, n_steps=40, n= 1):
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