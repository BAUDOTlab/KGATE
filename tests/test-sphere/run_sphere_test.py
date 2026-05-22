import pandas as pd
from kgate import Architect
from kgate import KnowledgeGraph


config_path = "/mnt/DATA_11TB/projects/knowledge_graph_embedding/kgate_dev/KGATE/tests/test-sphere/config_sphere_test.toml"

print(config_path)

# Path to split datasets
# Need to transform them into KnowledgeGraph objects to be usable
df_train = pd.read_csv("/mnt/DATA_11TB/projects/knowledge_graph_embedding/kgate_dev/KGATE/tests/test-sphere/FB-train.txt",
                       header = None,
                       names = ["head","edge","tail"])
kg_train = KnowledgeGraph(dataframe = df_train)

df_validation = pd.read_csv("/mnt/DATA_11TB/projects/knowledge_graph_embedding/kgate_dev/KGATE/tests/test-sphere/FB-valid.txt",
                       header = None,
                       names = ["head","edge","tail"])
kg_validation = KnowledgeGraph(dataframe = df_validation)

df_test = pd.read_csv("/mnt/DATA_11TB/projects/knowledge_graph_embedding/kgate_dev/KGATE/tests/test-sphere/FB-test.txt",
                       header = None,
                       names = ["head","edge","tail"])
kg_test = KnowledgeGraph(dataframe = df_test)


architect = Architect(config_path = config_path, kg = (kg_train, kg_validation, kg_test))

# Train the model using KG and hyperparameters specified in the configuration
architect.train_model()

# Test the trained model, using the best checkpoint
a = architect.test()

print(a)

# Run KG completion task, the empty list is the element that will be predicted
# known_heads = []
# known_relations = []
# known_tails = []
# architect.infer(known_heads, known_relations, known_tails)