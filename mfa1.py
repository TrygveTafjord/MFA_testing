import torch

class MFA(torch.nn.Module):
    def __init__(self, m_max, n_channels, k_max, tol=1e-4, max_iter=100, device='cpu'):
        super().__init__()
        self.m_max = m_max
        self.n_channels = n_channels
        self.k_max = k_max
        self.tol = tol
        self.max_iter = max_iter

        # pi_j: adaptable mixing proportions [cite: 87]
        self.register_buffer('pi', torch.empty(m_max)) 
        # mu_j: means for each component [cite: 85]
        self.register_buffer('mu', torch.empty(m_max, n_channels)) 
        # Lambda_j: factor loadings per component (m, p, k) 
        self.register_buffer('Lambda_j', torch.empty(m_max, n_channels, k_max)) 
        # Psi: shared diagonal noise variances [cite: 44, 108]
        self.register_buffer('Psi', torch.empty(n_channels)) 

        self.to(device)

    def _e_step(self, X):
        n_samples = X.shape[0]
        log_h = torch.zeros(n_samples, self.m_max, device=X.device)
        
        e_z = [] 
        e_zz = []

        # Shared diagonal noise matrix [cite: 44]
        diag_psi = torch.diag(self.Psi) + torch.eye(self.n_channels, device=X.device) * 1e-6

        for j in range(self.m_max):
            # Sigma_j = Lambda_j @ Lambda_j.T + Psi [cite: 46, 91]
            Sigma_j = self.Lambda_j[j] @ self.Lambda_j[j].T + diag_psi
            dist = torch.distributions.MultivariateNormal(self.mu[j], Sigma_j)
            
            # Equation (12): h_ij proportional to pi_j * N(x) [cite: 91]
            log_h[:, j] = torch.log(self.pi[j] + 1e-10) + dist.log_prob(X)

            # beta_j = Lambda' * (Psi + Lambda*Lambda')^-1 [cite: 54, 100]
            beta_j = self.Lambda_j[j].T @ torch.inverse(Sigma_j)
            
            diff = X - self.mu[j]
            # E[z | x, omega] [cite: 94]
            ez_j = diff @ beta_j.T 
            # E[zz' | x, omega] [cite: 101]
            ez_outer = torch.einsum('ni,nj->nij', ez_j, ez_j)
            term1 = torch.eye(self.k_max, device=X.device) - beta_j @ self.Lambda_j[j]
            ezz_j = term1.unsqueeze(0) + ez_outer
            
            e_z.append(ez_j)
            e_zz.append(ezz_j)

        responsibilities = torch.softmax(log_h, dim=1) # h_ij [cite: 91, 106]
        return responsibilities, e_z, e_zz

    def _m_step(self, X, responsibilities, e_z, e_zz):
        n_samples = X.shape[0]
        self.pi = responsibilities.mean(dim=0) # Equation (16) [cite: 169]

        new_Psi_sum = torch.zeros_like(self.Psi)

        for j in range(self.m_max):
            hj = responsibilities[:, j]
            sum_hj = hj.sum() + 1e-10

            # Augmented factor approach for Lambda and Mu [cite: 151, 157]
            ez_j = e_z[j]
            ezz_j = e_zz[j]

            # Numerator: Sum(h * x * E[z_tilde]') [cite: 157]
            term_x_ez = (hj.unsqueeze(1) * X).T @ ez_j
            term_x_1 = (hj.unsqueeze(1) * X).sum(dim=0, keepdim=True).T
            numerator = torch.cat([term_x_ez, term_x_1], dim=1)

            # Denominator: Sum(h * E[z_tilde @ z_tilde']) [cite: 161]
            sum_h_ezz = (hj.view(-1, 1, 1) * ezz_j).sum(dim=0)
            sum_h_ez = (hj.unsqueeze(1) * ez_j).sum(dim=0, keepdim=True)
            
            top_row = torch.cat([sum_h_ezz, sum_h_ez.T], dim=1)
            bot_row = torch.cat([sum_h_ez, torch.tensor([[sum_hj.item()]], device=X.device)], dim=1)
            denominator = torch.cat([top_row, bot_row], dim=0)

            # Solve for new augmented Lambda [cite: 157]
            lambda_tilde_new = numerator @ torch.inverse(denominator + torch.eye(self.k_max + 1, device=X.device) * 1e-6)
            self.Lambda_j[j] = lambda_tilde_new[:, :self.k_max]
            self.mu[j] = lambda_tilde_new[:, self.k_max]

            # Update shared Psi (diagonal constraint) [cite: 164]
            ez_tilde = torch.cat([ez_j, torch.ones(n_samples, 1, device=X.device)], dim=1)
            recon = ez_tilde @ lambda_tilde_new.T
            new_Psi_sum += (hj.unsqueeze(1) * (X - recon) * X).sum(dim=0)

        self.Psi = torch.abs(new_Psi_sum / n_samples) + 1e-5

    def fit(self, X):
        self._initialize_parameters(X)
        prev_ll = -float('inf')
        
        for i in range(self.max_iter):
            # Fixed Scoping: pass E-step outputs to M-step
            responsibilities, e_z, e_zz = self._e_step(X)
            self._m_step(X, responsibilities, e_z, e_zz)
            
            current_ll = self._compute_log_likelihood(X).item()
            if abs(current_ll - prev_ll) < self.tol:
                print(f"Converged at iteration {i}. LL: {current_ll:.4f}")
                break
            prev_ll = current_ll
            if i % 10 == 0: print(f"Iter {i}, LL: {current_ll:.4f}")

    def _compute_log_likelihood(self, X):
        # Implementation of Equation (7) using log-sum-exp [cite: 72]
        n_samples = X.shape[0]
        log_probs = torch.zeros(n_samples, self.m_max, device=X.device)
        for j in range(self.m_max):
            Sigma_j = self.Lambda_j[j] @ self.Lambda_j[j].T + torch.diag(self.Psi) + torch.eye(self.n_channels, device=X.device) * 1e-6
            dist = torch.distributions.MultivariateNormal(self.mu[j], Sigma_j)
            log_probs[:, j] = torch.log(self.pi[j] + 1e-10) + dist.log_prob(X)
        return torch.logsumexp(log_probs, dim=1).sum()

    def _initialize_parameters(self, X):
        n_samples, _ = X.shape
        self.pi.data.fill_(1.0 / self.m_max)
        # Randomly pick points for initial means [cite: 84]
        indices = torch.randperm(n_samples)[:self.m_max]
        self.mu.data.copy_(X[indices])
        self.Lambda_j.data.normal_(0, 0.1)
        self.Psi.data.copy_(X.var(dim=0) + 1e-2)