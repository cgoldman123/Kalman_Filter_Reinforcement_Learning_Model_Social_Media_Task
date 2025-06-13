import numpy as np
from scipy.stats import norm, qmc

def jsd_normal(mu1, sigma1, mu2, sigma2, n_samples=30_000, rng=None, eps=1e-12):
    """
    Jensen–Shannon divergence between two univariate Gaussians, each
    specified by a scalar mean (mu) and *standard deviation* (sigma).

    Parameters
    ----------
    mu1, sigma1 : float
        Mean and std-dev of the first Gaussian  N(mu1, sigma1**2).
    mu2, sigma2 : float
        Mean and std-dev of the second Gaussian N(mu2, sigma2**2).
    n_samples : int, optional
        Monte-Carlo sample size (default 30 000 gives ~1e-4 precision).
    rng : np.random.Generator or None
        Supply your own RNG for reproducibility (default: fresh PCG64).
    eps : float
        Safety constant to avoid log(0) when the pdf underflows.

    Returns
    -------
    jsd : float
        Jensen–Shannon divergence in *nats*.
    """
    rng = np.random.default_rng(rng)

    # -------- 1. draw antithetic (balanced around mean) and low-discrepancy (evenly-spaced) samples from P and Q  -------------
    # Sobol sequence is used to generate a between 0 and 1 that is more-evenly spaced than true random samples.
    sobol = qmc.Sobol(d=1, scramble=True, seed=rng)
    # Draw 2^m samples from the sequence, where m is the smallest integer such that 2^m >= n_samples.
    u = sobol.random_base2(int(np.ceil(np.log2(n_samples))))
    # Treat the samples between 0 and 1 as percentiles of the normal distribution, allowing us to get antithetic points centered around 0.
    z = norm.ppf(u.ravel())                        # N(0,1) → antithetic points

    x_p = mu1 + sigma1 * z[:n_samples]            # Take samples from P
    x_q = mu2 + sigma2 * z[:n_samples]            # Take samples from Q

    # -------- 2. evaluate densities at each of the sampled points ------------------------------------
    p_xp = norm.pdf(x_p, mu1, sigma1) + eps
    q_xp = norm.pdf(x_p, mu2, sigma2) + eps
    m_xp = 0.5 * (p_xp + q_xp)

    p_xq = norm.pdf(x_q, mu1, sigma1) + eps
    q_xq = norm.pdf(x_q, mu2, sigma2) + eps
    m_xq = 0.5 * (p_xq + q_xq)

    # -------- 3. KL parts & JSD ----------------------------------------
    kl_p = np.mean(np.log(p_xp) - np.log(m_xp))
    kl_q = np.mean(np.log(q_xq) - np.log(m_xq))
    return 0.5 * (kl_p + kl_q)

# mu1, sigma1 = 50, 8          # N(0,1)
# mu2, sigma2 = 84, 8          # N(2,0.8²)

# jsd_val = jsd_normal(mu1, sigma1, mu2, sigma2)
# print(f"JSD = {jsd_val:.6f} nats  ({jsd_val/np.log(2):.6f} bits)")
# print()


