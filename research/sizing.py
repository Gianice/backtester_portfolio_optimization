
# research/sizing.py
import numpy as np
import cvxpy as cp


def size_portfolio(signals, B):
    """
    Turn signal directions into factor-neutral weights.

    signals : np.ndarray, shape (n,)
        +1 long, -1 short, 0 flat, one entry per stock.
    B : np.ndarray, shape (n, K)
        Betas: B[i, k] is stock i's loading on factor k.

    Returns
    -------
    w : np.ndarray, shape (n,)
        Weights with sum(|w|) = 1, B.T @ w ~ 0, signs matching signals.
        All zeros if the problem is infeasible or nothing is active.
    status : str
        The solver status, e.g. "optimal" or "infeasible".
    """
    n = len(signals)
    signals = np.asarray(signals)
    B = np.asarray(B)

    # 1. Edge case: no active names.
    # TODO: if every signal is 0, return zeros(n) and a status string.
    if np.all(signals == 0):
        return np.zeros(n), 'no active'
    
    
    # 2. Build the starting weights w0.
    # TODO: w0 = signals, then scale so that sum(|w0|) == 1.
    #       Equal size per active name, long positive, short negative.
    
    w0 = signals/np.sum(np.abs(signals))
    
    

    # 3. Index sets, used for the sign constraints.
    # TODO: long_idx, short_idx, flat_idx from signals.
    long_idx = np.flatnonzero(signals > 0) #flatnonzero returns positions
    short_idx = np.flatnonzero(signals < 0)
    flat_idx = np.flatnonzero(signals == 0)
    
    

    # 4. Declare the variable.
    # TODO: 
    w = cp.Variable(n)

    # 5. Objective: stay close to w0.
    objective = cp.Minimize(cp.sum_squares(w - w0))

    # 6. Constraints.
    # TODO: factor neutrality  -> 
    constraints = [ B.T @ w == 0, w[long_idx] >=0, w[short_idx] <= 0, w[flat_idx] == 0]
    

    # 7. Solve.
    prob = cp.Problem(objective, constraints)
    prob.solve()

    # 8. Handle failure.
    # TODO: if prob.status is not "optimal": return zeros(n) and prob.status
    #       (change this rule if you decide on something else, and say why here)
    if prob.status != 'optimal':
        return np.zeros(n), prob.status
    
    
    # 9. Rescale so that sum(|w|) == 1 (only if the solution isn't ~zero).
    gross = np.sum(np.abs(w.value))
    if gross < 1e-6:
        return np.zeros(n), "degenerate"
    new_w = w.value / gross
    
    

    return new_w, prob.status