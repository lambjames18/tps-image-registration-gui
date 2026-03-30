import torch
from torch import nn
from torch.nn import functional as F


class TokenConfidence(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.token = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """get confidence tokens"""
        return (
            self.token(desc0.detach().float()).squeeze(-1),
            self.token(desc1.detach().float()).squeeze(-1),
        )


def sigmoid_log_double_softmax(
    sim: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor
) -> torch.Tensor:
    """create the log assignment matrix from logits and similarity"""
    b, m, n = sim.shape
    m0, m1 = torch.sigmoid(z0), torch.sigmoid(z1)
    certainties = torch.log(m0) + torch.log(m1).transpose(1, 2)
    scores0 = F.log_softmax(sim, 2)
    scores1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
    scores = scores0 + scores1 + certainties
    # scores[:, :-1, -1] = F.logsigmoid(-z0.squeeze(-1))
    # scores[:, -1, :-1] = F.logsigmoid(-z1.squeeze(-1))
    return scores, m0, m1


class MatchAssignment(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)

    # @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """build assignment matrix from descriptors"""
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d**0.25, mdesc1 / d**0.25
        sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores, m0, m1 = sigmoid_log_double_softmax(sim, z0, z1)
        return scores, sim, m0, m1

    def scores(self, desc0: torch.Tensor, desc1: torch.Tensor):
        m0 = torch.sigmoid(self.matchability(desc0)).squeeze(-1)
        m1 = torch.sigmoid(self.matchability(desc1)).squeeze(-1)
        return m0, m1


def filter_matches(scores: torch.Tensor, th: float):
    """obtain matches from a log assignment matrix [Bx M+1 x N+1]"""
    max0, max1 = scores.max(2), scores.max(1)
    m0, m1 = max0.indices, max1.indices
    indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
    indices1 = torch.arange(m1.shape[1], device=m1.device)[None]
    mutual0 = indices0 == m1.gather(1, m0)
    mutual1 = indices1 == m0.gather(1, m1)
    max0_exp = max0.values.exp()
    zero = max0_exp.new_tensor(0)
    mscores0 = torch.where(mutual0, max0_exp, zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
    if th is not None:
        valid0 = mutual0 & (mscores0 > th)
    else:
        valid0 = mutual0
    valid1 = mutual1 & valid0.gather(1, m1)
    m0 = torch.where(valid0, m0, -1)
    m1 = torch.where(valid1, m1, -1)
    return m0, m1, mscores0, mscores1
