import torch
import pytest
from kgate.decoders import TransE

@pytest.fixture
def transe_model():
    return TransE(emb_dim=50, n_entities=100, n_relations=10, dissimilarity_type='L2')

def test_initialization(transe_model):
    assert transe_model.emb_dim == 50
    assert transe_model.n_entities == 100
    assert transe_model.n_relations == 10

def test_score_function(transe_model):
    h_norm = torch.randn(32, 50)  # Batch size 32, embedding dim 50
    r_emb = torch.randn(32, 50)
    t_norm = torch.randn(32, 50)
    score = transe_model.score(h_norm=h_norm, r_emb=r_emb, t_norm=t_norm)
    assert score.shape == (32,)
    assert torch.is_tensor(score)

def test_get_embeddings(transe_model):
    assert transe_model.get_embeddigs() is None
