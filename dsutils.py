"""
Decoding similarity helper functions.
"""

import torch
import warnings
import numpy as np
import numpy.typing as npt
from typing import Tuple
from sklearn.model_selection import KFold
from tqdm import tqdm  # optional for progress bar
from itertools import product
from scipy.optimize import brentq
from torch import nn
# from torchvision import models


def rbf_kernel(X, Y=None, gamma=1.0, center=False):
    """
    Evaluate the RBF (Gaussian) kernel between two sets of vectors.

    Parameters:
    ----------
    X : ndarray of shape (n_samples_X, n_features)
    Y : ndarray of shape (n_samples_Y, n_features), optional
        If None, computes the kernel between X and itself.
    gamma : float
        Kernel coefficient (1 / (2 * sigma^2)).

    Returns:
    -------
    K : ndarray of shape (n_samples_X, n_samples_Y)
        RBF kernel matrix.
    """
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y) if Y is not None else X
    M = X.shape[0]
    Mp = Y.shape[0]

    # Squared Euclidean distance between each pair
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
    dist_sq = X_norm + Y_norm - 2 * np.dot(X, Y.T)

    # RBF kernel matrix
    K = np.exp(-gamma * dist_sq)

    if center:
        C = np.eye(M) - (1/M)*np.ones((M,M))
        Cp = np.eye(Mp) - (1/Mp)*np.ones((Mp,Mp))
        K = C@K@Cp
        return K
    else:  
        return K


def linear_kernel(X, Y=None, center=False):
    """
    Evaluate the linear kernel between two sets of vectors.

    Parameters:
    ----------
    X : ndarray of shape (n_samples_X, n_features)
    Y : ndarray of shape (n_samples_Y, n_features), optional
        If None, computes the kernel between X and itself.
    Returns:
    -------
    K : ndarray of shape (n_samples_X, n_samples_Y)
        linear kernel matrix.
    """
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y) if Y is not None else X
    M = X.shape[0]
    Mp = Y.shape[0]
    N = X.shape[1]

    # linear kernel matrix
    K = X@(Y.T)
    if center:
        C = np.eye(M) - (1/M)*np.ones((M,M))
        Cp = np.eye(Mp) - (1/Mp)*np.ones((Mp,Mp))
        K = C@K@Cp
        return K
    else:  
        return K

# measure R^2
def r2_score(y_true, y_pred):
    """
    Compute the R^2 (coefficient of determination) score between two vectors.
    y_true: ground truth vector
    y_pred: predicted vector
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

def solve_lambda_for_df(K, target_df):
    eigvals = np.linalg.eigvalsh(K)
    eigvals = np.maximum(eigvals, 0.0)

    def df(lam):
        return np.sum(eigvals / (eigvals + lam))

    # Find lam_hi such that df(lam_hi) < target_df
    lam_lo = 0.0
    lam_hi = 1e-12 + eigvals.max()
    while df(lam_hi) > target_df:
        lam_hi *= 10.0
        if lam_hi > 1e12 * (eigvals.max() + 1e-12):
            break

    return brentq(lambda lam: df(lam) - target_df, 1e-15, lam_hi)

def generalized_eig_psd(Q, B, k=10, tol=1e-12):
    """
    Solve Q a = mu B a for PSD (possibly singular) B by restricting to range(B).
    Returns smallest k generalized eigenpairs (mu, A) with B-orthonormal A.
    """
    Q = 0.5 * (Q + Q.T)
    B = 0.5 * (B + B.T)

    # Eigendecompose B
    b, U = np.linalg.eigh(B)
    b = np.maximum(b, 0.0)

    # Keep range(B)
    bmax = b.max() if b.size else 0.0
    keep = b > (tol * max(bmax, 1.0))
    if not np.any(keep):
        raise ValueError("B has (numerically) zero rank; cannot define test_var constraint.")

    U_r = U[:, keep]
    b_r = b[keep]

    # Whitening operator W: n x r
    W = U_r @ np.diag(1.0 / np.sqrt(b_r))

    # Reduced eigenproblem in r-dim
    C = W.T @ Q @ W
    C = 0.5 * (C + C.T)

    mu_all, V = np.linalg.eigh(C)  # ascending
    k = min(k, V.shape[1])

    mu = mu_all[:k]
    A = W @ V[:, :k]               # generalized eigenvectors in original coords

    # B-orthonormalize: A^T B A = I (should already hold; enforce numerically)
    for j in range(k):
        norm = np.sqrt(A[:, j].T @ B @ A[:, j])
        if norm > 0:
            A[:, j] /= norm

    return mu, A


def most_decodable_targets_krr(
    X_train,
    X_test,
    gamma,
    lam,
    target_df=0.1,
    n_targets=5,
    kernel_func=rbf_kernel,
    constraint="test_var",     # "test_var" or "rkhs_norm"
    center_test=True,          # used when constraint="test_var"
    rkhs_norm=1.0,             # desired ||f||_H (not squared) when constraint="rkhs_norm"
    reg_R=1e-10,               # stabilizer added to constraint matrix (R or K)
    return_alpha=True,
):
    """
    Compute 'most decodable' targets for fixed (gamma, lam) in KRR by solving:

        Q a = mu B a

    where Q = (K* - K*(K+lam I)^{-1}K)^T (K* - K*(K+lam I)^{-1}K),

    and the constraint matrix B is:
      - B = K*^T H K*  (if constraint="test_var", H centers test outputs if center_test=True)
      - B = K          (if constraint="rkhs_norm", which corresponds to fixing ||f||_H^2 = a^T K a)

    The smallest generalized eigenvalues correspond to the most decodable directions
    under the chosen constraint. Returned targets are noise-free:
        y_train = K a
        y_test  = K* a (optionally centered when constraint="test_var" and center_test=True)

    Notes:
      - If constraint="rkhs_norm", returned alpha vectors are scaled so that
            alpha^T K alpha = rkhs_norm^2
        i.e. ||f||_H = rkhs_norm in the RKHS induced by K.
      - If constraint="test_var", returned alpha vectors are scaled so that
            alpha^T B alpha = 1
        (unit centered test variance in kernel-span coordinates).
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    mu = X_train.mean(axis=0, keepdims=True)
    X_train = X_train - mu
    X_test = X_test - mu

    n = X_train.shape[0]
    m = X_test.shape[0]

    # Kernel matrices
    K = kernel_func(X_train, None, gamma=gamma)        # (n, n)
    Kstar = kernel_func(X_test, X_train, gamma=gamma)  # (m, n)

    # Set lam based on the df formula for KRR: df = trace(K (K + lam I)^{-1})
    df = target_df * n  # target effective degrees of freedom (heuristic)
    if lam is None:
        lam = solve_lambda_for_df(K, df)
    print(f"Using lambda={lam:.4e} to achieve target df={df:.1f}")

    # Build A = K* (K + lam I)^{-1} without forming inverse explicitly
    A_T = np.linalg.solve(K + lam * np.eye(n), Kstar.T)  # (n, m)
    A = A_T.T                                            # (m, n)

    # M = A K = K* (K+lam I)^{-1} K
    M = A @ K                                            # (m, n)

    # Q = (K* - M)^T (K* - M)
    D = Kstar - M                                        # (m, n)
    Q = D.T @ D                                          # (n, n)
    Q = 0.5 * (Q + Q.T)

    # Constraint matrix B
    if constraint == "test_var":
        if center_test:
            H = np.eye(m) - np.ones((m, m)) / m
            B = Kstar.T @ H @ Kstar
        else:
            B = Kstar.T @ Kstar
    elif constraint == "rkhs_norm":
        B = K
    else:
        raise ValueError("constraint must be 'test_var' or 'rkhs_norm'")

    B = 0.5 * (B + B.T)

    # Regularize B for numerical stability
    B_reg = (B + reg_R * np.eye(n))

    # Solve generalized eigenproblem via whitening:
    # B_reg^{-1/2} Q B_reg^{-1/2} v = mu v, alpha = B_reg^{-1/2} v
    # evals_B, U_B = np.linalg.eigh(B_reg)
    # if np.any(evals_B <= 0):
    #     raise ValueError(
    #         "Constraint matrix B is not positive definite even after reg_R; "
    #         "increase reg_R or change constraint/centering."
    #     )

    # inv_sqrt_B = U_B @ np.diag(1.0 / np.sqrt(evals_B)) @ U_B.T
    # C = inv_sqrt_B @ Q @ inv_sqrt_B
    # C = 0.5 * (C + C.T)

    # mu_all, V = np.linalg.eigh(C)  # ascending
    k = min(n_targets, n)

    # mu = mu_all[:k]
    # alpha = inv_sqrt_B @ V[:, :k]  # generalized eigenvectors
    mu, alpha = generalized_eig_psd(Q, B, k=n_targets, tol=1e-12)

    # Normalize according to the chosen constraint
    if constraint == "test_var":
        # alpha^T B_reg alpha = 1
        for j in range(k):
            denom = np.sqrt(alpha[:, j].T @ B_reg @ alpha[:, j])
            if denom > 0:
                alpha[:, j] /= denom
    else:  # constraint == "rkhs_norm"
        # enforce alpha^T K alpha = rkhs_norm^2 (use K, not B_reg, for the RKHS norm)
        target_sq = float(rkhs_norm) ** 2
        for j in range(k):
            denom_sq = alpha[:, j].T @ K @ alpha[:, j]
            if denom_sq <= 0:
                raise ValueError("Encountered non-positive RKHS norm; check reg_R or data scaling.")
            alpha[:, j] *= np.sqrt(target_sq / denom_sq)

    # # normalize alpha vectors to have unit norm in original space for interpretability
    # for j in range(k):
    #     norm = np.linalg.norm(alpha[:, j])
    #     if norm > 0:
    #         alpha[:, j] /= norm

    # Construct corresponding noise-free targets
    y_train = K @ alpha
    y_test = Kstar @ alpha

    # # normalize each target to have unit variance across test points for interpretability (optional, used when constraint="test_var")
    # if constraint == "test_var" and center_test:
    #     for j in range(k):
    #         var = np.var(y_test[:, j], ddof=1)
    #         if var > 0:
    #             y_test[:, j] /= np.sqrt(var)

    y_pred = A @ y_train

    if constraint == "test_var" and center_test:
        y_test = y_test - y_test.mean(axis=0, keepdims=True)

    rel_D = np.linalg.norm(D, 'fro') / np.linalg.norm(Kstar, 'fro')
    print("rel_D:", rel_D)

    out = {"mu": mu, "y_train": y_train, "y_test": y_test, "y_pred": y_pred, "Q": Q, "B": B, "K": K, "Kstar": Kstar, "lam": lam}
    if return_alpha:
        out["alpha"] = alpha
    return out





def cross_val_score_custom(model_class, X, Z, param_grid, loss_fn, cv=5, kernel='linear'):
    
    param_combos = list(product(*param_grid.values()))

    best_score = float('inf')
    best_params = None
    
    # Dictionary to store all parameters tried and their coefficients
    all_params_coefs = {}

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    for params in tqdm(param_combos):
        param_dict = dict(zip(param_grid.keys(), params))
        losses = []

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            Z_train, Z_val = Z[train_idx], Z[val_idx]

            # Initialize and fit the model
            probe = model_class(**param_dict, kernel = kernel, center_columns=True, fit_intercept=False)
            probe.fit(X_train, Z_train)

            Z_pred = probe.predict(X_val)
            loss = loss_fn(Z_val, Z_pred, params)
            losses.append(loss)

            all_params_coefs[tuple(params)] = probe.coef_

        avg_loss = np.mean(losses)
        if avg_loss < best_score:
            best_score = avg_loss
            best_params = param_dict

    return best_params, best_score, X_train, Z_train  #, all_params_coefs


def mse_loss(y_true, y_pred, params):
    # Example: weighted MSE (or any custom logic)
    M = y_true.shape[0]
    return np.mean((y_true - y_pred) ** 2)

def inner_product_loss(y_true, y_pred, params):
    # Example: weighted MSE (or any custom logic)
    M = y_true.shape[0]
    a = params[0]
    return np.mean((y_true - y_pred) ** 2)
    # return -np.trace(2* y_pred.T @ y_true - a* y_pred.T @ y_pred) #- y_true.T @ y_true)


class genKernelRegression:
    
    def __init__(self,  center_columns=True, kernel = 'linear',  a=0, b=1, gamma = 1.0, fit_intercept=False):
        self.center_columns = center_columns
        self.a = a
        self.b = b
        self.gamma = gamma
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.kernel = kernel


    def _add_intercept(self, X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def fit(self, X, Z):
        X = np.asarray(X)
        Z = np.asarray(Z)

        if self.center_columns:
            X = X - np.mean(X, axis=0)
            self.Z_mean_ = np.mean(Z, axis=0)
            Z = Z - self.Z_mean_
        else:
            self.Z_mean_ = 0.0

        if self.fit_intercept:
            X = self._add_intercept(X)

        n_features = X.shape[1]
        # I = np.eye(n_features)
        Im = np.eye(X.shape[0])
        
        if self.fit_intercept:
            I[0, 0] = 0  # do not regularize intercept

        # Closed-form kernel regression solution:

        if self.kernel == 'rbf':
            KX = rbf_kernel(X, gamma = self.gamma, center=True)
        elif self.kernel == 'linear':
            KX = linear_kernel(X,center=True)
        else: 
            KX = linear_kernel(X,center=True)
            
        # XtX = X.T @ X
        # Xty = X.T @ y\
        # print(self.b)
        # print('  ')
        # print(self.gamma)
        try:
            # self.weights_ = np.linalg.inv(self.a*KX + self.b *Im) @ Z
            # self.weights_ = np.linalg.pinv(self.a*KX + self.b *Im) @ Z
            inv_reg = np.linalg.solve(self.a*KX + self.b *Im, Im)
            self.weights_ = inv_reg @ Z
            H = KX @ inv_reg
            self.effective_dof_ = np.trace(H)

        except np.linalg.LinAlgError:
            print(f"Skipping hyperparams {self.a}, {self.b}, {self.gamma}: matrix is singular.")
            self.weights_ = np.full(Z.shape, np.nan)
            self.effective_dof_ = np.nan
        if self.fit_intercept:
            self.intercept_ = self.weights_[0, 0]
            self.coef_ = self.weights_[1:, 0]
        else:
            self.intercept_ = 0.0
            self.coef_ = self.weights_

        self.Xtrain = X
        
        return self

    def predict(self, X):
        X = np.asarray(X)
        M = X.shape[0]

        if self.kernel == 'rbf':
            KXXtrain = rbf_kernel(X,self.Xtrain, gamma = self.gamma, center=True)
        elif self.kernel == 'linear':
            KXXtrain = linear_kernel(X,self.Xtrain, center=True)
        else: 
            KXXtrain = linear_kernel(X,self.Xtrain, center=True)
        # KXXtrain = rbf_kernel(X,self.Xtrain, gamma = self.gamma)
        if self.fit_intercept:
            return self.intercept_ + KXXtrain @ self.coef_ + self.Z_mean_
        else:
            return KXXtrain @ self.coef_ + self.Z_mean_

    def score(self, X, Z):
        """R² score"""
        Z_pred = self.predict(X)
        ss_res = np.sum((Z - Z_pred) ** 2)
        ss_tot = np.sum((Z - np.mean(Z,axis=0)) ** 2)
        return 1 - ss_res / ss_tot

    def effective_dof(self, a=None, b=None, gamma=None):
        """Effective degrees of freedom for kernel ridge regression.

        Computes tr(K_X (a * K_X + b * I)^{-1}), which is the trace of the
        smoother (hat) matrix H that maps training targets to fitted values.

        When called with the same hyperparameters used during fit (the
        default), returns the value cached by fit() to avoid redundant
        computation.  Pass different a, b, or gamma to recompute.

        Parameters
        ----------
        a : float, optional
            Kernel weight. Defaults to self.a.
        b : float, optional
            Ridge penalty. Defaults to self.b.
        gamma : float, optional
            RBF kernel bandwidth (only used when kernel='rbf').
            Defaults to self.gamma.

        Returns
        -------
        dof : float
            Effective degrees of freedom (trace of the hat matrix).
        """
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        if gamma is None:
            gamma = self.gamma

        # Return cached value when hyperparameters match those used in fit()
        if a == self.a and b == self.b and gamma == self.gamma:
            return self.effective_dof_

        X = self.Xtrain

        if self.kernel == 'rbf':
            KX = rbf_kernel(X, gamma=gamma, center=True)
        else:
            KX = linear_kernel(X, center=True)

        n = KX.shape[0]
        Im = np.eye(n)
        H = KX @ np.linalg.solve(a * KX + b * Im, Im)
        return np.trace(H)

    def rkhs_norm(self, gamma=None):
        """RKHS norm of the fitted function fhat.

        Computes ||fhat||_H^2 = alpha^T K_X alpha, where alpha are the
        representer weights obtained from fit() and K_X is the kernel
        matrix evaluated on the training data.

        Parameters
        ----------
        gamma : float, optional
            RBF kernel bandwidth (only used when kernel='rbf').
            Defaults to self.gamma.

        Returns
        -------
        norm_sq : float or ndarray
            Squared RKHS norm.  Scalar when the target Z was single-output;
            array of shape (n_targets,) for multi-output regression.
        """
        if gamma is None:
            gamma = self.gamma

        X = self.Xtrain

        if self.kernel == 'rbf':
            KX = rbf_kernel(X, gamma=gamma, center=True)
        else:
            KX = linear_kernel(X, center=True)

        alpha = self.weights_
        # alpha^T K_X alpha; works for both single- and multi-output
        norm_sq = np.einsum('ij,ik,kj->j', alpha, KX, alpha)
        return norm_sq.item() if norm_sq.size == 1 else norm_sq



class Frechet:  # NEEDS REVISITING
    
    def __init__(self):
        pass

    # def mean(self, Cz, cached):
    #     Kbar = np.zeros(np.shape(cached[0]))
    #     nmodels = len(cached)
    #     d = np.shape(cached[0])[0]
    #     for i in range(d):
    #         for j in range(d):
    #             for n in range(nmodels):
    #                 M = cached[n]@Cz
    #                 Kbar[i,j] = Kbar[i,j] + (1/(nmodels*Cz[i,j]))*M[j,i]
    #         print(i)
    #     return Kbar


    # def mean(self, Cz, cached):
    #     Kbar = np.zeros(np.shape(cached[0]))
    #     nmodels = len(cached)
    #     d = np.shape(cached[0])[0]
    #     for n in range(nmodels):
    #         M = cached[n]@Cz
    #         Kbar = Kbar + (1/nmodels) * (1/Cz) * np.transpose(M)
    #         print(n)
    #     return Kbar
    
    def arith_mean(self, cached):
        Abar = np.zeros(np.shape(cached[0]))
        nmodels = len(cached)
        d = np.shape(cached[0])[0]
        for n in range(nmodels):
            Abar = Abar + (1/nmodels) * cached[n]
            print(n)
        return Abar

    def mean(self, cached, Cz):
        Abar = np.zeros(np.shape(cached[0]))
        nmodels = len(cached)
        d = np.shape(cached[0])[0]
        B = (2/nmodels)*np.sum(Cz@cached,axis=0)
        Lambda, U = np.linalg.eig(Cz)
        UBU = np.transpose(U) @ B @ U
        D = np.zeros(np.shape(Cz))
        for i in range(d):
            for k in range(d):
                D[i,k] = (1/(Lambda[i] + Lambda[k]))*UBU[i,k]
        Kbar = U @ D @ np.transpose(U)
        return Kbar


    # def variance(self, Cz, cached, Kbar):
    #     fvar = 0
    #     for i in range(len(cached)):
    #         fvar = fvar + np.trace(Kbar@Kbar@Cz) + np.trace(cached[i]@cached[i]@Cz) - 2 *np.trace(Kbar@cached[i]@Cz)
    #         print(i)
    #     return fvar

    def variance(self, cached, Kbar, Cz):
        fvar = 0
        for i in range(len(cached)):
            # fvar = fvar + 2 - 2 *np.trace(np.conjugate(np.transpose(Abar))@cached[i])
            fvar = fvar + np.trace(Kbar@Cz@Kbar) + np.trace(cached[i]@Cz@cached[i]) - 2*np.trace(Kbar@Cz@cached[i])
            print(i)
        return fvar
    


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

class SoftMaxModule(nn.Module):
    def __init__(self):
        super(SoftMaxModule, self).__init__()
        self.softmax = nn.Softmax()

    def forward(self, x):
        return self.softmax(x)


def bespoke_cov_matrix(z):
    """
    Computes the covariance matrix for a custom set of tasks.  Assumes tasks are equally weighted.  

    Parameters
    ----------
    z : list of tasks.  Each element of list should be an array of (number of samples x 1).  These are the desired readouts for every input sample.  

    Returns
    -------
    Cz : ndarray
        M x M empirical covariance matrix.
    """

    n = len(z)

    Cz = 0
    for i in range(n):
        Cz = Cz + z[i]@z[i].T
    Cz = Cz/(n)

    return Cz




def random_partitions_cov_matrix(M, n):
    """
    Computes the empirical task covariance matrix for a set of n random binary partitions of the input samples.  Readouts are sampled from Rademacher distribution for each input sample. 

    Parameters
    ----------
    M : int (number of input samples)
    n : int (number of random tasks to generate)

    Returns
    -------
    Cz : ndarray
        M x M empirical covariance matrix.
    """
    z = []
    for i in range(n):
        zrand = np.random.randint(0,2,(M,1)) 
        zrand = 2*zrand - 1
        z.append(zrand)

    Cz = 0
    for i in range(n):
        Cz = Cz + z[i]@z[i].T
    Cz = Cz/(n)

    return Cz


def gaussian_partitions_cov_matrix(M, n):
    """
    Computes the empirical task covariance matrix if the desired readouts are a set of n samples from the M-dimensional standard normal distribution. 

    Parameters
    ----------
    M : int (number of input samples)
    n : int (number of random tasks to generate)

    Returns
    -------
    Cz : ndarray
        M x M empirical covariance matrix.
    """
    z = []
    for i in range(n):
        zrand = np.random.normal(0,1,(M,1)) 
        z.append(zrand)

    Cz = 0
    for i in range(n):
        Cz = Cz + z[i]@z[i].T
    Cz = Cz/(n)

    return Cz


def update_gaussian_cov_matrix(Cz, n, delta_n):
    """ Update a Gaussian covariance matrix generated by gaussian_partitions_cov_matrix(M, n) with delta_n more samples.  
    """
    M = np.shape(Cz)[0]
    new_cov_contrib = delta_n*gaussian_partitions_cov_matrix(M,delta_n)
    newCz = (1/(n + delta_n))*(n*Cz + new_cov_contrib)
    return newCz

def update_random_cov_matrix(Cz, n, delta_n):
    """ Update a random binary partitions covariance matrix generated by random_partitions_cov_matrix(M, n) with delta_n more samples.  
    """
    M = np.shape(Cz)[0]
    new_cov_contrib = delta_n*random_partitions_cov_matrix(M,delta_n)
    newCz = (1/(n + delta_n))*(n*Cz + new_cov_contrib)
    return newCz


class PartitionsCovMatrix:
    """
    Make task covariance matrices to evaluate average decoding similarity.  
    """
    def __init__(self, M, n_initial, method = 'binary'):
        self.M = M
        self.n = n_initial
        self.method = method
        self.matrix = None

    def initialize_cov_matrix(self):
        if self.method == 'binary':
            Cz = random_partitions_cov_matrix(self.M, self.n)
        elif self.method == 'gaussian':
            Cz = gaussian_partitions_cov_matrix(self.M, self.n)
        else:
            raise ValueError(
                "method must be either 'binary' or 'gaussian'.")
        self.matrix = Cz
        
        return None

    def update_cov_matrix(self, add_n):
        if self.method == 'binary':
            Cz = update_random_cov_matrix(self.matrix, self.n, add_n)
        elif self.method == 'gaussian':
            Cz = update_gaussian_cov_matrix(self.matrix, self.n, add_n)
        else:
            raise ValueError(
                "method must be either 'binary' or 'gaussian'.")
        
        self.n = self.n + add_n
        self.matrix = Cz
        
        return None


def whiten(
    X: npt.NDArray, 
    alpha: float, 
    preserve_variance: bool = True, 
    eigval_tol=1e-7
    ) -> Tuple[npt.NDArray, npt.NDArray]:
    """Return regularized whitening transform for a matrix X.

    Parameters
    ----------
    X : ndarray
        Matrix with shape `(m, n)` holding `m` observations
        in `n`-dimensional feature space. Columns of `X` are
        expected to be mean-centered so that `X.T @ X` is
        the covariance matrix.
    alpha : float
        Regularization parameter, `0 <= alpha <= 1`. When
        `alpha == 0`, the data matrix is fully whitened.
        When `alpha == 1` the data matrix is not transformed
        (`Z == eye(X.shape[1])`).
    preserve_variance : bool
        If True, rescale the (partial) whitening matrix so
        that the total variance, trace(X.T @ X), is preserved.
    eigval_tol : float
        Eigenvalues of covariance matrix are clipped to this
        minimum value.

    Returns
    -------
    X_whitened : ndarray
        Transformed data matrix.
    Z : ndarray
        Matrix implementing the whitening transformation.
        `X_whitened = X @ Z`.
    """

    # Return early if regularization is maximal (no whitening).
    if alpha > (1 - eigval_tol):
        return X, np.eye(X.shape[1])

    # Compute eigendecomposition of covariance matrix
    lam, V = np.linalg.eigh(X.T @ X)
    lam = np.maximum(lam, eigval_tol)

    # Compute diagonal of (partial) whitening matrix.
    # 
    # When (alpha == 1), then (d == ones).
    # When (alpha == 0), then (d == 1 / sqrt(lam)).
    d = alpha + (1 - alpha) * lam ** (-1 / 2)

    # Rescale the whitening matrix.
    if preserve_variance:

        # Compute the variance of the transformed data.
        #
        # When (alpha == 1), then new_var = sum(lam)
        # When (alpha == 0), then new_var = len(lam)
        new_var = np.sum(
            (alpha ** 2) * lam
            + 2 * alpha * (1 - alpha) * (lam ** 0.5)
            + ((1 - alpha) ** 2) * np.ones_like(lam)
        )

        # Now re-scale d so that the variance of (X @ Z)
        # will equal the original variance of X.
        d *= np.sqrt(np.sum(lam) / new_var)

    # Form (partial) whitening matrix.
    Z = (V * d[None, :]) @ V.T

    # An alternative regularization strategy would be:
    #
    # lam, V = np.linalg.eigh(X.T @ X)
    # d = lam ** (-(1 - alpha) / 2)
    # Z = (V * d[None, :]) @ V.T

    # Returned (partially) whitened data and whitening matrix.
    return X @ Z, Z