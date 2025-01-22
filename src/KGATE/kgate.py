import os
from pathlib import Path
from torchkge import Model
import torch
from .utils import parse_config

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
        torch.set_num_threads(num_cores)

        # Create output folder if it doesn't exist
        os.makedirs(config["output_directory"], exist_ok=True)

        run_kg_prep =  config["run_kg_preprocess"]
        run_training = config["run_training"]
        run_eval = config["run_evaluation"]

        if run_kg_prep:
            kg_train, kg_val, kg_test = prepare_knowledge_graph(config)