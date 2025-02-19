import torch
import pytest

def test_score_function(transe_model):
    h_norm = torch.randn(32, 50)  # Batch size 32, embedding dim 50
    r_emb = torch.randn(32, 50)
    t_norm = torch.randn(32, 50)
    score = transe_model.score(h_norm=h_norm, r_emb=r_emb, t_norm=t_norm)
    assert score.shape == (32,)
    assert torch.is_tensor(score)

