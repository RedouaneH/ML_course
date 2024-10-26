import numpy as np

def compute_subgradient_mae(y, tx, w):
    """Compute a subgradient of the MAE at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the subgradient of the MAE at w.
    """

    N = y.shape[0]
    
    def ae_subgradient(x):
        if x < 0:
            return -1
        if x > 0:
            return 1
        return 0
         

    ae_subgradient_vectorized = np.vectorize(ae_subgradient)

    return np.sum(-tx * ae_subgradient_vectorized(y - tx @ w)[:, np.newaxis], axis = 0) * (1/N)
