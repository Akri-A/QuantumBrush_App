import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import equinox as eqx
import pennylane as qml

"""
Utility functions
"""
def quantum_fidelity(psi, rho):
    psi = psi/jnp.linalg.norm(psi)
    rho = rho /jnp.linalg.norm(rho)
    return jnp.abs(jnp.vdot(psi, rho))**2

"""
Hamiltonian builder
"""
def build_hamiltonians(n_qubits, key = jr.PRNGKey(0)): 
    """
    Build Hamiltonian Ansatz
    """
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

"""
Circuit builders
"""
def build_splitting_circuit(dev, H0, H1, n_qubits):
    """
    dev: backend
    """
    @qml.qnode(dev)
    def splitting_circuit(model, initial_state, T, n_steps, n=1):
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
    
    # for better precision, could eventually be a user_defined parameter
    jax.config.update("jax_enable_x64", True)
    
    # Build Ansatz
    key = jr.PRNGKey(0)
    key, Hkey = jr.split(key)
    H0, H1 = build_hamiltonians(n_qubits, Hkey)

    # Souce and target state preparation
    key, inkey, outkey = jr.split(key, 3)
    initial_state = jr.normal(inkey, shape=(2**n_qubits))
    target_state = jr.normal(outkey, shape=(2**n_qubits))
    initial_state /= jnp.linalg.norm(initial_state)
    target_state /= jnp.linalg.norm(target_state)

    n_epochs = params.get("n_epochs", 1000)
    n_steps = params.get("timesteps", 40)
    T = params.get("max T", 1.0)
    lr = params.get("lr", 0.02)

    key, mlpkey = jax.random.split(key)

    model = eqx.nn.MLP(
        in_size='scalar', out_size='scalar', depth=2, width_size=32, activation=jax.nn.tanh, key=mlpkey
    )
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Build circuit for training
    dev = qml.device("default.qubit", wires=n_qubits)
    circuit = build_splitting_circuit(dev, H0, H1, n_qubits)

    @eqx.filter_jit
    def loss_fn(model, inital_state, target_state, T=1.0, n_steps=40, C=0):#3e-4):
        psi = circuit(model, inital_state, T, n_steps)
        fidelity = quantum_fidelity(psi, target_state)
        ## 
        ts = jnp.linspace(0, T, n_steps)
        integral = jax.scipy.integrate.trapezoid(jax.vmap(model)(ts)**2, ts)
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
    print(f"Final fidelity: {quantum_fidelity(rho_f, target_state)}")

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
        qml.StatePrep(initial_state, wires=range(n_qubits))
        for k in range(n_steps):
            t_k = k * dt
            u_k = model(jnp.array(t_k))
            # Strang-splitting time step
            qml.ApproxTimeEvolution(H0, dt/2, n)
            qml.ApproxTimeEvolution(u_k * H1, dt, n)
            qml.ApproxTimeEvolution(H0, dt/2, n)
        return qml.state()
    print("=== Circuit built. ===")
    return final_circuit



