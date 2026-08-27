# Separators that can be used to load a CSV file
SUPPORTED_SEPARATORS = [",","\t",";"]

# Builtin KGATE initializers
SUPPORTED_INITIALIZERS = [
    "Random",
    "Feature",
    "Node2Vec"
]

# Builtin KGATE encoders
SUPPORTED_ENCODERS = [
    "Default",
    "GCN",
    "GAT"
]

# Builtin KGATE decoders
SUPPORTED_DECODERS = [
    "TransE",
    "TransH",
    "TransR",
    "TransD",
    "TorusE",
    "RESCAL",
    "DistMult",
    "ComplEx",
    "ConvKB"
]

# Builtin KGATE negative samplers
SUPPORTED_SAMPLERS = [
    "Positional",
    "Uniform",
    "Bernoulli",
    "Mixed"
]
