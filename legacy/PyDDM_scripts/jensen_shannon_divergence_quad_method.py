import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
 
def jsd_normal(mu1, sigma1, mu2, sigma2, *,
                       base=np.e, eps=1e-300, limit=100):
    """
    Jensen–Shannon divergence (in nats by default) between two 1-D normals.
   
    Parameters
    ----------
    mu1, sigma1 : float
        Mean and standard deviation of the first distribution 𝒩(μ₁, σ₁²).
    mu2, sigma2 : float
        Mean and standard deviation of the second distribution 𝒩(μ₂, σ₂²).
    base : float, optional
        Logarithm base for the output (default e).  Use base=2 for bits.
    eps : float, optional
        Floor added to each pdf to avoid log-underflow in extreme tails.
    limit : int, optional
        Integration subdivision limit passed to `scipy.integrate.quad`.
   
    Returns
    -------
    js : float
        Jensen–Shannon divergence D_JS(P‖Q) in the chosen log base.
    """
    if sigma1 <= 0 or sigma2 <= 0:
        raise ValueError("Standard deviations must be positive.")
 
    p = norm(loc=mu1, scale=sigma1)
    q = norm(loc=mu2, scale=sigma2)
 
    # Core integrand: ½·p·log(p/m) + ½·q·log(q/m)
    def integrand(x):
        p_x = p.pdf(x) + eps
        q_x = q.pdf(x) + eps
        m_x = 0.5 * (p_x + q_x)
        return 0.5 * (p_x * np.log(p_x / m_x) +
                      q_x * np.log(q_x / m_x))
 
    js, _ = quad(integrand, -np.inf, np.inf, limit=limit)
    if base != np.e:          # convert from nats to requested base
        js /= np.log(base)
    return js
 
 
# print(jsd_normal(50, 8, 42, 8))
# print("hi")