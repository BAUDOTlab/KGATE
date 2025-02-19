import pytest
import pandas as pd
from kgate.decoders import TransE
from kgate import KGATEGraph

@pytest.fixture
def toy_kg_df():
    data = {
        "from": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B","A", "B", "C", "D", "A", "B", "C", "D"],
        "rel": ["r", "r", "r", "r", "r","r", "r", "r", "r", "r", "r2", "r2", "r2", "r2", "r2","r2", "r2", "r2"],
        "to": ["B", "C", "D", "E", "F", "C", "D", "E", "F", "G", "E", "E", "E", "E", "F", "F", "F", "F"]
    }
    return pd.DataFrame(data)

@pytest.fixture
def toy_mappings():
    mappings = {
        "id": ["A", "B", "C", "D", "E","F", "G"],
        "type": ["AB","AB","CD","CD", "EF","EF", "AB"]
    }
    return pd.DataFrame(mappings)

@pytest.fixture
def toy_kg(toy_kg_df):
    return KGATEGraph(df=toy_kg_df)

@pytest.fixture
def transe_model():
    return TransE(emb_dim=50, n_entities=100, n_relations=10, dissimilarity_type='L2')

