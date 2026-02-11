import torch

class MFA(torch.nn.Module):
    def __init__(self, m_max, n_channels, k_max, tol=1e-4, max_iter=100, device='cpu'):
        super().__init__()

        # x = lambda * z + u
        self.register_buffer('m', torch.empty(m_max)) # number of components (FA)
        self.register_buffer('k', torch.empty(m_max, k_max)) # number of factors
        self.register_buffer('Psi', torch.empty(n_channels)) # noise variances, the diagonal of the covariance matrix, we assume the noise to be the same across all components (sensor noise)
        self.register_buffer('lambda', torch.empty(n_channels, k_max)) # factor loadings
        self.register_buffer('mu', torch.empty(m_max, n_channels)) # means for each component (FA)
        self.register_buffer('pi', torch.empty(m_max)) # adaptable mixing proportions, pi_i = P(omega) (FA) (tthe probability of a pixel belonging to a specific FA component)


        self.m_max = m_max
        self.n_channels = n_channels
        self.k_max = k_max

    @property
    def device(self):
        return self.device_tracker.device
