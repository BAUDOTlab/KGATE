import os
import logging
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from kgate.knowledgegraph import KnowledgeGraph

logging.basicConfig(
    level = logging.INFO,  
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def get_data_root_directory() -> Path:
    """Gets or create the root directory of KGATE data.

    Looks first for the `KGATE_DATA_ROOT` environment variable, and fallback to
    the current working directory if it is not found.

    Returns
    -------
    **data_root_directory** *pathlib.Path*
    : The root directory of KGATE data
    """
    root_directory = Path(os.environ.get("KGATE_DATA_ROOT", Path().cwd().joinpath("KGATE_DATA")))

    root_directory.mkdir(parents = True, exist_ok = True)
    return root_directory

def load_FB15k_237(data_directory: os.PathLike = None, keep_split: bool = False) -> KnowledgeGraph:
    """
    Load the knowledge graph FB15k-237 into memory.

    If the data is not on disk, first download it.

    Arguments
    ---------
    **data_directory** *(os.PathLike, optional)*
    : Directory to read or download the knowledge graph to.

    **keep_split** *(bool)*
    : Whether to keep the original split between training, validation and test set.
    As the original split contains a significant part of data leakage, it is recommended
    to set this option to True only to compare with existing results.

    Returns
    -------
    **knowledge_graph** *(KnowledgeGraph)*
    The FB15k-237 knowledge graph.
    """
    logging.info("Loading FB15k-237 dataset into memory...")
    output_directory: Path = data_directory or get_data_root_directory().joinpath("FB15k-237")

    output_directory.mkdir(exist_ok = True)

    freebase_urls = ["https://github.com/villmow/datasets_knowledge_embedding/blob/master/FB15k-237/train.txt",
                     "https://github.com/villmow/datasets_knowledge_embedding/blob/master/FB15k-237/valid.txt",
                     "https://github.com/villmow/datasets_knowledge_embedding/blob/master/FB15k-237/test.txt"]

    for url in freebase_urls:
        filename = url.split["/"][:-1]
        filepath = output_directory.joinpath(filename)
        if not filepath.exists():
            logging.info(f"Downloading {filename} from {url}...")
            urlretrieve(url, filepath)

    training_df = pd.read_csv(output_directory.joinpath("train.txt"), sep="\t", header=None, names=["head","edge","tail"])
    validation_df = pd.read_csv(output_directory.joinpath("valid.txt"), sep="\t", header=None, names=["head","edge","tail"])
    test_df = pd.read_csv(output_directory.joinpath("test.txt"), sep="\t", header=None, names=["head","edge","tail"])

    knowledge_graph_df = pd.concat([training_df, validation_df, test_df])
    knowledge_graph = KnowledgeGraph(dataframe = knowledge_graph_df)

    if keep_split:
        knowledge_graph.train_mask[:len(training_df)] = True
        knowledge_graph.validation_mask[len(training_df) : len(training_df) + len(validation_df)] = True
        knowledge_graph.test_mask[len(training_df) + len(validation_df):] = True

    logging.info("Dataset successfully loaded.")

    return knowledge_graph

def load_WN18RR(data_directory: os.PathLike = None, keep_split: bool = False) -> KnowledgeGraph:
    """
    Load the knowledge graph WN18RR into memory.

    If the data is not on disk, first download it.

    Arguments
    ---------
    **data_directory** *(os.PathLike, optional)*
    : Directory to read or download the knowledge graph to.

    **keep_split** *(bool)*
    : Whether to keep the original split between training, validation and test set.
    As the original split contains a significant part of data leakage, it is recommended
    to set this option to True only to compare with existing results.

    Returns
    -------
    **knowledge_graph** *(KnowledgeGraph)*
    The WN18RR knowledge graph.
    """
    logging.info("Loading WN18RR dataset into memory...")
    output_directory: Path = data_directory or get_data_root_directory().joinpath("WN18RR")

    output_directory.mkdir(exist_ok = True)

    freebase_urls = ["https://github.com/villmow/datasets_knowledge_embedding/blob/master/WN18RR/text/train.txt",
                     "https://github.com/villmow/datasets_knowledge_embedding/blob/master/WN18RR/text/valid.txt",
                     "https://github.com/villmow/datasets_knowledge_embedding/blob/master/WN18RR/text/test.txt"]

    for url in freebase_urls:
        filename = url.split["/"][:-1]
        filepath = output_directory.joinpath(filename)
        if not filepath.exists():
            logging.info(f"Downloading {filename} from {url}...")
            urlretrieve(url, filepath)

    training_df = pd.read_csv(output_directory.joinpath("train.txt"), sep="\t", header=None, names=["head","edge","tail"])
    validation_df = pd.read_csv(output_directory.joinpath("valid.txt"), sep="\t", header=None, names=["head","edge","tail"])
    test_df = pd.read_csv(output_directory.joinpath("test.txt"), sep="\t", header=None, names=["head","edge","tail"])

    knowledge_graph_df = pd.concat([training_df, validation_df, test_df])
    knowledge_graph = KnowledgeGraph(dataframe = knowledge_graph_df)

    if keep_split:
        knowledge_graph.train_mask[:len(training_df)] = True
        knowledge_graph.validation_mask[len(training_df) : len(training_df) + len(validation_df)] = True
        knowledge_graph.test_mask[len(training_df) + len(validation_df):] = True

    logging.info("Dataset successfully loaded.")
    
    return knowledge_graph

def load_PrimeKG():
    pass