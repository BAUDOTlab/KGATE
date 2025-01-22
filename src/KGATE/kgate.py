import os
from pathlib import Path
from torchkge import Model
import torch
from .utils import parse_config, load_knowledge_graph, set_random_seeds
from .preprocessing import prepare_knowledge_graph
from .encoders import *
from .decoders import *
from torchkge.utils import MarginLoss, BinaryCrossEntropyLoss, DataLoader
import logging

# Configure logging
logging.captureWarnings(True)
log_level = logging.INFO# if config["common"]['verbose'] else logging.WARNING
logging.basicConfig(
    level=log_level,  
    format='%(asctime)s - %(levelname)s - %(message)s' 
)

TRANSLATIONAL_MODELS = ['TransE', 'TransH', 'TransR', 'TransD', 'TorusE']


# Architect
class KGATE(Model):
    def __init__(self, kg, device, config_path: str = "", cudnn_benchmark = True, num_cores = 0, **kwargs):
        
        config = parse_config(config_path, kwargs)

        if torch.cuda.is_available():
            # Benchmark convolution algorithms to chose the optimal one.
            # Initialization is slightly longer when it is enabled.
            torch.backends.cudnn.benchmark = cudnn_benchmark

        # If given, restrict the parallelisation to user-defined threads.
        # Otherwise, use all the cores the process has access to.
        num_cores = num_cores if num_cores > 0 else len(os.sched_getaffinity(0))
        logging.info(f"Setting number of threads to {num_cores}")
        torch.set_num_threads(num_cores)

        # Create output folder if it doesn't exist
        logging.info(f"Output folder: {config["output_directory"]}")
        os.makedirs(config["output_directory"], exist_ok=True)

        run_kg_prep =  config["run_kg_preprocess"]
        run_training = config["run_training"]
        run_eval = config["run_evaluation"]

        if run_kg_prep:
            logging.info(f"Preparing KG...")
            self.kg_train, self.kg_val, self.kg_test = prepare_knowledge_graph(config)
            if not run_training and not run_eval:
                logging.info("KG preprocessed.")
                return
        else:
            logging.info("Loading KG...")
            self.kg_train, self.kg_val, self.kg_test = load_knowledge_graph(config["kg_pkl"])
            logging.info("Done")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Detected device: {device}')

        set_random_seeds(config["seed"])

        self.emb_dim = self.config["model"]["emb_dim"]
        self.rel_emb_dim = self.config["model"]["rel_emb_dim"]
        logging.info("Initializing encoder...")
        self.initialize_encoder()

        logging.info("Initializing decoder...")
        self.initialize_decoder()

    def initialize_encoder(self):
        mapping_csv = self.config["mapping_csv"]
        encoder_config = self.config["model"]["encoder"]
        encoder_name = encoder_config["name"]
        gnn_layers = encoder_config["gnn_layer_number"]

        match encoder_name:
            case "Default":
                self.encoder = DefaultEncoder(self.kg_train.n_ent, self.kg_train.n_rel, self.emb_dim)

        

    def initialize_decoder(self):
        decoder_config = self.config["model"]["decoder"]
        decoder_name = decoder_config["name"]
        dissimilarity = decoder_config["dissimilarity"]
        margin = decoder_config["margin"]

        # Translational models
        match decoder_name:
            case "TransE":
                self.decoder = TransE(self.emb_dim, self.kg_train.n_ent, self.kg_train.n_rel,
                               dissimilarity_type=dissimilarity)
                self.crietion = MarginLoss(margin)
