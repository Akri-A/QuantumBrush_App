'''
Author: Chih-Kang Huang && chih-kang.huang@hotmail.com
Date: 2025-11-11 15:45:26
LastEditors: Chih-Kang Huang && chih-kang.huang@hotmail.com
LastEditTime: 2025-11-13 22:55:33
FilePath: /steerable/steer_training.py
Description: 


'''

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import equinox as eqx
import pennylane as qml

import importlib.util

spec_helper = importlib.util.spec_from_file_location("helper", "effect/steerable/helper.py")
helper = importlib.util.module_from_spec(spec_helper)
spec_helper.loader.exec_module(helper)

spec_models = importlib.util.spec_from_file_location("models ", "effect/steerable/models.py")
models = importlib.util.module_from_spec(spec_models)
spec_models.loader.exec_module(models )

#jax.config.update("jax_enable_x64", True)


"""
Circuit builders
"""
def build_splitting_circuit(dev, H_list, n_qubits):
    """
    dev: backend
    """
    @qml.qnode(dev)
    def splitting_circuit(model, initial_state, T, n_steps, n=1):
        """ 
        model: control NN
        initial_state: Initial Quantum State
        H_list : Hamiltanians
        T: final time
        n_steps: time steps
        n: trotterizaiton order
        """
        dt = T / n_steps
        H0 = H_list[0]
        qml.StatePrep(initial_state, wires=range(n_qubits))
        for k in range(n_steps):
            t_k = k * dt
            u_k = model(jnp.array(t_k))
            # Strang-splitting time step
            qml.ApproxTimeEvolution(H0, dt/2, 1)
            for u, H in zip(list(u_k), H_list[1:]): 
                qml.ApproxTimeEvolution(u*H, dt/2, 1)
            for u, H in (zip(reversed(list(u_k)), reversed(H_list[1:]))): 
                qml.ApproxTimeEvolution(u*H, dt/2, 1)
            qml.ApproxTimeEvolution(H0, dt/2, 1)
        return qml.state()
    return splitting_circuit

def build_circuit(backend, params, source, target, n_qubits):

    if source.shape != target.shape :
        raise ValueError(
            f"source and target must have the same size "
            f"(got source={len(source)}, target={len(target)})"
        )
    
    if len(source) != 2 ** n_qubits:
        raise ValueError(
            f"Number of parameters must be 2**{n_qubits} = {2**n_qubits}, "
            f"but got {len(source)}."
        )
    
    
    key = jr.PRNGKey(0)

    # Build Ansatz
    H_list = helper.build_hamiltonians(n_qubits)

    # Souce and target state preparation
    initial_state = source 
    target_state = target
    initial_state /= jnp.linalg.norm(initial_state)
    target_state /= jnp.linalg.norm(target_state)

    n_epochs = params.get("n_epochs", 500)
    n_steps = params.get("timesteps", 30)
    T = params.get("max T", 1.0)
    lr = params.get("lr", 0.05)

    key, mlpkey = jax.random.split(key)

    model = eqx.nn.MLP(
        in_size='scalar', out_size=len(H_list)-1, depth=2, width_size=16, activation=jax.nn.tanh, key=mlpkey
    )
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Build circuit for training
    dev = qml.device("default.qubit", wires=n_qubits)
    circuit = build_splitting_circuit(dev, H_list, n_qubits)

    @eqx.filter_jit
    def loss_fn(model, inital_state, target_state, T=1.0, n_steps=40, C=1e-5):#3e-4):
        psi = circuit(model, inital_state, T, n_steps)
        fidelity = helper.quantum_fidelity(psi, target_state)
        ## 
        ts = jnp.linspace(0, T, n_steps)
        integral = jax.scipy.integrate.trapezoid(jax.vmap(lambda t : jnp.linalg.norm(model(t))**2)(ts), ts)
        return 1 - fidelity + C*integral

    @eqx.filter_jit
    def make_step(model, opt_state, initial_state, target_state, T=1.0, n_steps=40):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, initial_state, target_state, T, n_steps)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    # Training
    print(f"=== Start traing ===")
    for step in range(n_epochs):
        model, opt_state, loss = make_step(model, opt_state, initial_state, target_state, T, n_steps)
        if step % (n_epochs // 10) == 0:
            print(f"Step {step:03d}: loss = {loss:.6f}")
    
    rho_f = circuit(model, initial_state, T, n_steps)
    print(f"Final fidelity: {helper.quantum_fidelity(rho_f, target_state)}")

    @qml.qnode(backend)
    def final_circuit(initial_state, n_qubits, T, n_steps=40, n=1):
        """ 
        model: control NN
        initial_state: Initial Quantum State
        H0, H1: Hamiltanians
        T: final time
        n_steps: time steps
        n: trotterizaiton order
        """
        dt = T / n_steps
        H0 = H_list[0]
        qml.StatePrep(initial_state, wires=range(n_qubits))
        for k in range(n_steps):
            t_k = k * dt
            u_k = model(jnp.array(t_k))
            # Strang-splitting time step
            qml.ApproxTimeEvolution(H0, dt/2, 1)
            for u, H in zip(list(u_k), H_list[1:]): 
                qml.ApproxTimeEvolution(u*H, dt/2, 1)
            for u, H in (zip(reversed(list(u_k)), reversed(H_list[1:]))): 
                qml.ApproxTimeEvolution(u*H, dt/2, 1)
            qml.ApproxTimeEvolution(H0, dt/2, 1)
        return qml.state()
    print("=== Circuit built. ===")
    return final_circuit



