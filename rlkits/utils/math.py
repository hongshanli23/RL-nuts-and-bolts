import numpy as np
import torch
from torch.distributions import Normal


def explained_variance(ypred,y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary
        

def conjugate_gradient(f_Ax, b, cg_iters=10, verbose=False, residual_tol=1e-10):
    x = torch.zeros_like(b, dtype=torch.float32)
    
    # residual and A-conjugate basis
    r, p = b.clone(), b.clone()
    rr = torch.dot(r, r)
    
    fmtstr =  "%10i %10.3g %10.3g"
    titlestr =  "%10s %10s %10s"
    if verbose: print(titlestr % ("iter", "residual norm", "soln norm"))
        
    for i in range(cg_iters):
        if verbose: print(fmtstr % (i, rr, np.linalg.norm(x.numpy())))
        Ap = f_Ax(p)
        alpha = rr / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = torch.dot(r, r)
        beta = - rr_new / rr
        p = r - beta * p
        rr = rr_new
        if rr < residual_tol:
            print("residual tolerance achieved, good enough for now")
            break
    
    if verbose: print(fmtstr % (i+1, rr, np.linalg.norm(x.numpy())))
    return x
        

    
    
def _KL_normal(dist1:Normal, dist2:Normal):
    """KL of two normal distribution"""

    sigma1, sigma2 = dist1.stddev, dist2.stddev
    mu1, mu2 = dist1.mean, dist2.mean

    kl = (
        torch.log(sigma2 / sigma1)
        + (torch.pow(sigma1, 2) + torch.pow(mu2 - mu1, 2))
        / (2 * torch.pow(sigma2, 2))
            - 0.5
    )
    return kl

    
def KL(dist1, dist2):
    """compute KL(dist1 || dist2)"""
    if isinstance(dist1, Normal) and isinstance(dist2, Normal):
        return _KL_normal(dist1, dist2)
        