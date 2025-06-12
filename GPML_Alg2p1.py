# A Gaussian process is a collection of random variables, any # finite number of which have a joint
# Gaussian distribution. (p13)

# A practical implementation of Gaussian process regression (GPR) is shown in Algorithm 2.1.  The algorithm
# uses Cholesky decomposition, instead of directly inverting the matrix, since it is faster and numerically
# more stable.  The algorithm returns the predicitve mean and variance for noise-free test data. (p19)

import numpy as np

# A.4, R&W page 202
# https://en.wikipedia.org/wiki/Cholesky_decomposition
def cholesky(A):

    n = A.shape[0]
    L = np.zeros_like(A)

    for i in range(n):
        for j in range(i + 1):
            s = np.dot(L[i, :j], L[j, :j])
            if i == j:
                L[i, j] = np.sqrt(A[i, i] - s)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
    return L

# L\b: Solves Lx = b for x, where L is a lower triangular matrix, using forward substition.
# https://en.wikipedia.org/wiki/Triangular_matrix#Forward_and_back_substitution
def lmd(L, b): # Left Matrix Divide

    n = L.shape[0]
    x = np.zeros_like(b, dtype=float)

    for i in range(n):
        x[i] = (b[i] - np.dot(L[i, :i], x[:i])) / L[i, i]

    return x

# Algorithm 2.1, R&W page 19
def alg2p1 (X_inputs, y_targets, k_covarfunc, rho_noise, xs_testin):

    # K = K(X,X) is the n x n matrix of the covariances evaluated at all pairs of training outputs (p15)
    n = y_targets.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = k_covarfunc(y_targets[i], y_targets[j])

    # Step 2: Cholesky decomposition of the covariances plus the noise for numerical stability
    L = cholesky(K + rho_noise*np.eye(n))

    # Step 3: 
    alpha = lmd(L.T, lmd(L, y_targets))

    # The vector of covariances between the test points and the training points (p17)
    k_star = np.array([[k_covarfunc(xs, x) for x in X_inputs] for xs in xs_testin])

    # Step 4: predictive mean at the test points, eq 2.25, pg 17
    mean = np.dot(k_star, alpha)

    # Step 5:
    v = lmd(L, k_star)

    # The covariance matrix between the test points and the training points
    k_xsxs = np.array([[k_covarfunc(x1, x2) for x2 in xs_testin] for x1 in xs_testin])

    # Step 6: predictive variance at the test points, eq 2.26, pg 17
    variance = k_xsxs - np.dot(v.T, v)

    # Step 7: log marginal likelihood, eq 2.30, pg 19
    marginal_likelihood = (
        -0.5 * np.dot(y_targets.T, alpha)
        - np.sum(np.log(np.diag(L)))
        - 0.5 * n * np.log(2 * np.pi)
    )
    
    # Step 8: Round results to the nearest thousandth before returning
    mean = np.round(mean, 3)
    variance = np.round(variance, 3)
    marginal_likelihood = np.round(marginal_likelihood, 3)
    return mean, variance, marginal_likelihood


# Squared exponential covariance function (p14)
def squared_exponential(x1, x2, length_scale=1.0, sigma_f=1.0):
    return sigma_f**2 * np.exp(-0.5 * ((x1 - x2) ** 2) / length_scale**2)


def main():

    X_inputs = np.array([0.0, 1.0, 2.0])
    y_targets = np.array([1.0, 2.0, 0.5])
    xs_testin = np.array([1.5, 2.0, 2.5])

    print("Inputs:", X_inputs)
    print("Targets:", y_targets)
    print("Test Inputs:", xs_testin)
    print("Executing R&W Algorithm 2.1...")

    mean, variance, marginal_likelihood = alg2p1(
        X_inputs,
        y_targets,
        squared_exponential,
        rho_noise=0.1,
        xs_testin=xs_testin
    )

    print("Predictive mean: ", mean)
    print("Predictive variance:\n", variance)
    print("Log marginal likelihood: ", marginal_likelihood)

main()
