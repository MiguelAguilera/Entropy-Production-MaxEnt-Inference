import torch

def get_EP_gd(X, i, x0=None):
    N = X.shape[0]
    if x0 is None:
        x0 = np.zeros(N)

    g_t = torch.zeros_like(X)
    for j in range(N):
        g_t[j,:] = -2*X[i,:]*X[j,:]    # these are the observables
    g_t_noI = torch.cat((g_t[:i,:], g_t[i+1:,:]))
    g_avg=g_t_noI.mean(axis=1)           # conditional averages 

    def func(theta): 
        obj = theta@g_avg - torch.log(torch.exp(-theta @ g_t_noI).mean())
        #obj -= torch.logsumexp(vals, dim=0) - torch.log(torch.tensor(vals.numel()))
        return -obj  # minus because code is for gradient descent

    
    theta, v = gradient_descent(
        func,
        x0=x0.clone().detach(), # np.zeros(N-1),
        lr=.01,
        optimizer='Adam',
        tol=1e-4,
        # report_every=20,
        num_iters=100
    )
    return -v, theta

def adam_optimizer(objective_func, gradient_func, initial_params, learning_rate=0.01, 
                  beta1=0.9, beta2=0.999, epsilon=1e-8, tol=1e-3, max_iter=100):
# CLAUDE GAVE ME THIS IMPLEMENTATION. IT SHOULD BE CHECKED.
    """
    Adam optimization algorithm.
    
    Parameters:
    - objective_func: Function that computes the objective value
    - gradient_func: Function that computes the gradient
    - initial_params: Initial parameter values (numpy array)
    - learning_rate: Step size parameter (alpha)
    - beta1: Exponential decay rate for first moment estimates
    - beta2: Exponential decay rate for second moment estimates
    - epsilon: Small constant for numerical stability
    - tol: Tolerance for convergence
    - max_iter: Maximum number of iterations
    
    Returns:
    - params: Optimized parameters
    - objective_values: List of objective function values
    - iterations: Number of iterations until convergence
    """
    
    # Initialize parameters
    params = initial_params.clone()
    m = torch.zeros_like(params)  # First moment estimate
    v = torch.zeros_like(params)  # Second moment estimate
    objective_values = [objective_func(params)]
    t = 0
    
    # Run Adam until convergence or max iterations
    while t < max_iter:
        t += 1
        
        # Compute gradient
        grad = gradient_func(params)
        
        # Update biased first and second moment estimates
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        
        # Compute bias-corrected first and second moment estimates
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # Update parameters
        prev_params = params.clone()
        params = params - learning_rate * m_hat / (torch.sqrt(v_hat) + epsilon)
        
        # Calculate current objective value
        obj_val = objective_func(params)
        objective_values.append(obj_val)
        
        # Check for convergence
        param_change = torch.norm(params - prev_params)
        if param_change < tol:
            print(f"Converged after {t} iterations. Final param change: {param_change:.8f}")
            break
        print(t, obj_val)
            
    if t == max_iter:
        print(f"Reached maximum iterations ({max_iter}) without converging. Last param change: {param_change:.8f}")
        
    return params, objective_values, t


def gradient_descent(f, x0, optimizer='Adam', lr=0.1, num_iters=1000, tol=1e-5, report_every=None, opt_args={}):
    x = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
    opt_obj = getattr(torch.optim, optimizer)
    opt     = opt_obj([x], lr=lr, **opt_args)
    loss_history = []
    prev_loss = float('inf')
    for i in range(num_iters):
        opt.zero_grad()
        loss = f(x)
        loss.backward()
        opt.step()
        if report_every is not None and (i+1) % report_every == 0:
            print(f"Iteration {i+1}: {loss.item():.6f}")
        if abs(prev_loss - loss.item()) < tol:
            break
        prev_loss = loss.item()
    
    return x.detach(), loss.item()

if False:
    # Run gradient descent starting from x = 5
    final_x, last_loss = gradient_descent(
        objective_function, 
        initial_x=5.0,
        lr=learning_rate,
        num_iters=num_iterations
    )




from torch.utils.data import DataLoader, TensorDataset

def batch_gradient_descent(f, x0, data, optimizer='Adam', lr=0.1, num_iters=1000, tol=1e-5, report_every=None,
                      batch_size=32, num_epochs=100):
    x = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
    opt_obj = getattr(torch.optim, optimizer)
    opt     = opt_obj([x], lr=lr)

    dataset = TensorDataset(data.T)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    prev_loss = float('inf')
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in data_loader:
            batch_data = batch[0]
            opt.zero_grad()
            loss = f(x, batch_data.T)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            num_batches += 1
        avg_epoch_loss = epoch_loss / num_batches
        
        if report_every is not None and (epoch+1) % report_every == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.6f}")

        if abs(prev_loss - loss.item()) < tol:
            break
        prev_loss = loss.item()
    
    return x, loss.item()

