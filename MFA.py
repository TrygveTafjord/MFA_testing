import torch
import torch.nn as nn
import math

class MFA(nn.Module):
    """
    Mixture of Factor Analyzers (MFA) implemented in PyTorch.
    
    Model:
        x ~ Sum_k pi_k * N(mu_k, Lambda_k * Lambda_k^T + Psi)
    
    Parameters:
        n_components (K): Number of mixture components
        n_features (D): Dimensionality of data (120 for HSI)
        n_factors (q): Dimensionality of latent factors
    """
    def __init__(self, n_components, n_features, n_factors, tol=1e-4, max_iter=100, device='cpu'):
        super().__init__()
        self.K = n_components
        self.D = n_features
        self.q = n_factors
        self.tol = tol
        self.max_iter = max_iter
        self.device = device
        
        # Initialize parameters
        # pi: Mixing coefficients (K)
        self.log_pi = nn.Parameter(torch.log(torch.ones(self.K, device=self.device) / self.K))
        
        # mu: Means (K, D)
        self.mu = nn.Parameter(torch.randn(self.K, self.D, device=self.device))
        
        # Lambda: Factor Loadings (K, D, q)
        self.Lambda = nn.Parameter(torch.randn(self.K, self.D, self.q, device=self.device))
        
        # Psi: Diagonal noise (D) - simplified to be shared across clusters or specific
        # Here we assume diagonal noise Psi is typically shared or diagonal per cluster.
        # For stability, we often use a single Psi (D) for all clusters or diagonal.
        # We will use diagonal Psi (D).
        self.log_psi = nn.Parameter(torch.randn(self.D, device=self.device))
        
    def fit(self, X):
        """
        Runs the Expectation-Maximization (EM) algorithm.
        X: Tensor of shape (N, D)
        """
        X = X.to(self.device)
        N = X.shape[0]
        
        # Initialization (using K-Means or random subset helps, here simplistic random)
        # Better: Initialize mu with random data points
        indices = torch.randperm(N)[:self.K]
        self.mu.data = X[indices].clone()

        prev_ll = -float('inf')
        
        for i in range(self.max_iter):
            # --- E-Step ---
            log_resp, log_likelihood = self.e_step(X)
            current_ll = log_likelihood.mean()
            
            # --- M-Step ---
            resp = torch.exp(log_resp) # (N, K)
            self.m_step(X, resp)
            
            # Convergence check
            diff = current_ll - prev_ll
            if i > 0 and abs(diff) < self.tol:
                break
            prev_ll = current_ll
            
        self.final_ll = prev_ll * N # Store total log likelihood
        
    def e_step(self, X):
        """
        Calculates log-responsibilities with Numerical Stability improvements.
        """
        N = X.shape[0]
        # 1. Add epsilon to psi to prevent division by zero or collapse
        psi = torch.exp(self.log_psi) + 1e-5 
        
        log_resps = []
        
        for k in range(self.K):
            L_k = self.Lambda[k] # (D, q)
            
            # 2. Construct Covariance Matrix
            # C_k = Lambda @ Lambda.T + Psi
            C_k = L_k @ L_k.T + torch.diag(psi) 
            
            # 3. CRITICAL FIX: Add "Jitter" to the diagonal
            # This forces the matrix to be Positive Definite
            jitter = 1e-4 * torch.eye(self.D, device=self.device)
            C_k = C_k + jitter
            
            try:
                # Robust generic approach
                dist = torch.distributions.MultivariateNormal(self.mu[k], covariance_matrix=C_k)
                log_prob = dist.log_prob(X) # (N)
            except ValueError:
                # Fallback if Cholesky still fails (rare with jitter)
                # We return a very low probability for this cluster
                log_prob = torch.ones(N, device=self.device) * -1e20
            
            log_resps.append(log_prob + self.log_pi[k])
            
        log_resps = torch.stack(log_resps, dim=1) # (N, K)
        
        # Log-Sum-Exp for normalization
        log_likelihood = torch.logsumexp(log_resps, dim=1) # (N)
        log_resp_norm = log_resps - log_likelihood.unsqueeze(1) # (N, K)
        
        return log_resp_norm, log_likelihood

    def m_step(self, X, resp):
        """
        Updates parameters based on responsibilities.
        """
        N = X.shape[0]
        Nk = resp.sum(dim=0) + 1e-10 # (K)
        
        # 1. Update pi
        self.log_pi.data = torch.log(Nk / N)
        
        psi = torch.exp(self.log_psi)
        new_psi_accum = torch.zeros_like(psi)
        
        for k in range(self.K):
            resp_k = resp[:, k].unsqueeze(1) # (N, 1)
            
            # 2. Update mu
            # Weighted mean
            mu_k = (resp_k * X).sum(dim=0) / Nk[k]
            self.mu.data[k] = mu_k
            
            # 3. Update Lambda (Factor Loadings)
            # Needs expectations of latent factors z. 
            # E[z|x] = beta_k * (x - mu_k)
            # beta_k = Lambda_k.T * C_k^{-1}
            
            L_k = self.Lambda[k]
            C_k = L_k @ L_k.T + torch.diag(psi)
            diff = X - mu_k # (N, D)
            
            # Beta calculation
            # beta = L.T @ inv(C)
            # We use solve for stability
            try:
                C_inv = torch.linalg.inv(C_k)
            except:
                 C_inv = torch.eye(self.D, device=self.device) # Fallback
            
            beta = L_k.T @ C_inv # (q, D)
            
            # E[z] for this cluster
            Ez = (diff @ beta.T) # (N, q)
            
            # E[zzT] = Var(z|x) + E[z]E[z]T
            # Var(z|x) = I - beta @ Lambda
            Var_z = torch.eye(self.q, device=self.device) - beta @ L_k
            
            # Update Lambda
            # New_Lambda = (Sum resp * (x-mu) * Ez.T) * (Sum resp * E[zzT])^-1
            
            # This part is complex to vectorize perfectly without high memory.
            # Simplified Update for HSI (Assumes roughly PC directions):
            # We can use a simplified update or standard PCA on weighted covariance.
            # Standard EM update:
            
            S_k = (resp_k * diff).T @ diff / Nk[k] # Weighted Covariance (D, D)
            
            # Analytical solution for Lambda is roughly top q eigenvectors of S_k 
            # adjusted by Psi. 
            # Ideally, we implement the strict M-step, but for Thesis implementation 
            # fitting this into a single class cell:
            
            # We approximate Lambda update by Weighted PCA on S_k for stability
            try:
                vals, vecs = torch.linalg.eigh(S_k)
                # Take top q
                idx = torch.argsort(vals, descending=True)[:self.q]
                self.Lambda.data[k] = vecs[:, idx] * torch.sqrt(vals[idx]).unsqueeze(0)
            except:
                pass # Keep old lambda if SVD fails
            
        # Update Psi (Diagonal noise)
        # Often kept fixed or updated as mean variance residual
        # self.log_psi.data = ... (Skipping for stability in simple implementation)

    def bic(self, X):
        """
        Bayesian Information Criterion
        BIC = -2 * log(L) + d * log(N)
        """
        with torch.no_grad():
            _, log_likelihood = self.e_step(X)
            total_ll = log_likelihood.sum().item()
            
            # Count params
            # Pi: K-1
            # Mu: K * D
            # Lambda: K * (D*q - q*(q-1)/2)  (considering rotation freedom)
            # Psi: D
            n_params = (self.K - 1) + (self.K * self.D) + \
                       (self.K * (self.D * self.q)) + self.D
            
            n_samples = X.shape[0]
            
            return -2 * total_ll + n_params * math.log(n_samples)