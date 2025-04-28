import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.distributions.normal import Normal

class VPG(nn.Module):
    def __init__(self, obs_dim=6, act_dim=2, hl=[128, 128]) -> None:
        super(VPG, self).__init__()
        print(f"Initializing VPG Model: obs_dim={obs_dim}, act_dim={act_dim}, hidden_layers={hl}")

        layers = []
        layers.append(nn.Linear(obs_dim, hl[0]))
        layers.append(nn.ReLU())

        for i in range(len(hl) - 1):
            layers.append(nn.Linear(hl[i], hl[i+1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hl[-1], act_dim * 2))

        self.model = nn.Sequential(*layers)
        self.act_dim = act_dim
        self.std_offset=1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return logits

    def get_distribution(self, state: torch.Tensor) -> torch.distributions.Normal:
        logits = self.forward(state)
        mean, log_std = logits.chunk(2, dim=-1) 

        std = F.softplus(log_std) + self.std_offset
        # std = torch.clamp(F.softplus(log_std), min=1e-2, max=1.0)
        self.std_offset = self.std_offset * 0.99997
        # self.std_offset = max(1e-2, min(self.std_offset, 1.0))
        # print(mean)
        # print(std)
        # std = torch.exp(log_std) + 5e-2

        distrib = Normal(mean, std)
        return distrib
