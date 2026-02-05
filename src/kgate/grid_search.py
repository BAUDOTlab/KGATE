import logging
from typing import Tuple, Any

import optuna
import pandas as pd

from .architect import Architect
from .knowledgegraph import KnowledgeGraph
from .utils import parse_config


logging.captureWarnings(True)
logging_level = logging.INFO
logging.basicConfig(
    level = logging_level,  
    format = "%(asctime)s - %(levelname)s - %(message)s" 
)


def run_grid_search(config_path: str,
                    number_of_trials: int = 10,
                    kg: Tuple[KnowledgeGraph, KnowledgeGraph, KnowledgeGraph] 
                        | KnowledgeGraph 
                        | None = None,
                    dataframe: pd.DataFrame 
                                | None = None):
    """
    Run a grid search hyperparameter optimization according to the given configuration.

    To register a hyperparameter in the grid search optimization, set it as a list in the configuration.

    Not all hyperparameters can be evaluated. The full list is:
    TODO.hyperparameters_list
    
    Arguments
    ---------
    number_of_trials: int, default to 10
        TODO.What_that_argument_is_or_does
    kg: Tuple[KnowledgeGraph, KnowledgeGraph, KnowledgeGraph] or KnowledgeGraph, optional
        Knowledge graph on which a grid search hyperparameter optimization will be done.
    dataframe: pd.DataFrame, optional
        TODO.What_that_argument_is_or_does
    
    Notes
    -----
    If the configuration file has no hyperparameter list, this function is effectively the same
    as running `Architect(config_path).train_model()`
    
    """
    def objective(trial: optuna.trial.Trial):
        config = parse_config(  config_path = config_path,
                                config_dictionnary = {})

        config = {key: suggest_value(trial, key, config[key]) for key in config}
        
        architect = Architect(  kg = kg,
                                df = dataframe,
                                **config)

        architect.train_model()

        result = architect.test()
        
        return result["Global_metrics"]

    study = optuna.create_study(direction = "maximize",
                                pruner = optuna.pruners.MedianPruner())
    study.optimize( objective,
                    n_trials = number_of_trials)

    best_trial = study.best_trial
    logging.info(f"Best trial score: {best_trial.value}")
    logging.info(f"Best trial hyperparameters: ")
    for key, value in best_trial.params.items():
        logging.info("{}: {}".format(key, value))


def suggest_value(  trial: optuna.trial.Trial,
                    value_name: str,
                    value: int | float | list, # TODO check if the types are correct
                    ) -> int | float | list: # TODO check if the types are correct
    """
    TODO.What_the_function_does_about_globally
    
    Arguments
    ---------
    trial: optuna.trial.Trial
        TODO.What_that_argument_is_or_does
    value_name: str
        TODO.What_that_argument_is_or_does
    value: int or float or list
        TODO.What_that_argument_is_or_does
    
    Returns
    -------
    suggested_value: int or float or list
        The value suggested.
    
    """
    logging.info(value_name)
    logging.info(value)
    
    if value_name == "evaluation":
        return value
    
    elif isinstance(value, dict):
        return {child_key: suggest_value(trial, child_key, value[child_key]) for child_key in value}
    
    elif isinstance(value, list):
        if len(value) == 0:
            return value
        
        elif len(value) == 3 and (isinstance(value[0], int) or isinstance(value[0], float)):
            low, high = value[:2]
            step = None
            log = False
            if isinstance(value[2], bool):
                log = True
                
            else:
                step = value[2]
                
            match type(value[0]):
                case "float":
                    return trial.suggest_float( name = value_name, 
                                                low = low, 
                                                high = high, 
                                                step = step, 
                                                log = log)
                case "int":
                    return trial.suggest_int(   name = value_name,
                                                low = low,
                                                high = high,
                                                step = step,
                                                log = log)
        else:
            return trial.suggest_categorical(name = value_name, choices = value)
        
    else:
        return value