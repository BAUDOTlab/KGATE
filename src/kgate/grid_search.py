import optuna
from .kgate import Architect
from .data_structures import KGATEGraph
from .utils import parse_config
import pandas as pd
from torchkge import KnowledgeGraph
from typing import Tuple, Any

def run_grid_search(config_path: str, kg: Tuple[KGATEGraph,KGATEGraph,KGATEGraph] | KnowledgeGraph | None = None, df: pd.DataFrame | None = None):
    """Run a grid search hyperparameter optimization according to the given configuration.

    To register a hyperparameter in the grid search optimization, set it as a list in the configuration.

    Not all hyperparameters can be evaluated. The full list is:
    
    Notes
    -----
    If the configuration file has no hyperparameter list, this function is effectively the same
    as running `Architect(config_path).train_model()`"""


    def objective(trial: optuna.trial.Trial):
        config = parse_config(config_path=config_path, config_dict={})

        config = {key: suggest_value(trial, key, config[key]) for key in config}

        architect = Architect(config_path=config_path, kg=kg, df=df)

def suggest_value(trial: optuna.trial.Trial, value_name: str, value: Any):
    if value_name == "evaluation":
        return value
    elif isinstance(value, dict):
        return {child_key: suggest_value(trial, child_key, value[child_key]) for child_key in value}
    elif isinstance(value, list):
        if len(value) == 3 and (isinstance(value[0], int) or isinstance(value[0], float)):
            
            low, high = value[:2]
            step = None
            log = False
            if isinstance(value[2], bool):
                log = True
            else:
                step = value[2]

            match type(value[0]):
                case "float":
                    return trial.suggest_float(name = value_name, 
                                            low=low, 
                                            high=high, 
                                            step=step, 
                                            log=log)
                case "int":
                    return trial.suggest_int(name = value_name,
                                            low = low,
                                            high = high,
                                            step = step,
                                            log = log)
        else:
            return trial.suggest_categorical(name=value_name, choices=value)
    else:
        return value