import logging
from typing import Tuple, Any

import optuna
import pandas as pd

from .architect import Architect
from .knowledgegraph import KnowledgeGraph
from .config import Configuration

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

    To register a hyperparameter in the grid search optimization, set it as a list in the
    configuration (a `[low, high]` or `[low, high, step]` numeric range, a
    `[low, high, True]` log-scale range, or a list of categorical choices). The
    optimization is run with Optuna (`direction="maximize"`, median pruner),
    maximizing the `"Global_metrics"` result of `Architect.test()`.

    Arguments
    ---------
    
    **config_path** *(str)*
    : Path to the KGATE configuration file; hyperparameters to search over are
      specified as lists in this file.
    
    **number_of_trials** *(int, default to 10)*
    : Number of optimization trials run by Optuna.
    
    **kg** *(Tuple[KnowledgeGraph, KnowledgeGraph, KnowledgeGraph] or KnowledgeGraph, optional)*
    : Knowledge graph on which a grid search hyperparameter optimization will
      be done (a single KGATE knowledge graph, or a train/validation/test
      tuple), passed to each Architect.
    
    **dataframe** *(pd.DataFrame, optional)*
    : The knowledge graph as a pandas DataFrame, alternative to `kg`.
    
    Notes
    -----
    
    If the configuration file has no hyperparameter list, this function is
    effectively the same as running `Architect(config_path).train_model()`
    (repeated `number_of_trials` times, with the same hyperparameters).
    
    """
    def objective(trial: optuna.trial.Trial):
        config = Configuration(  config_path = config_path,
                                config_dict = {})

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
    Suggest a value for a single hyperparameter from an Optuna trial, based on
    how it is written in the configuration.

    Rules applied:
    - the `"evaluation"` key is never sampled (returned as is);
    - a `dict` is recursed into, key by key;
    - an empty list is returned as is;
    - a list of the form `[low, high]` or `[low, high, step]` (first element
      numeric) is sampled with `trial.suggest_float` or `trial.suggest_int`
      over `[low, high]` with the given step, or on a logarithmic scale when
      the third element is `True`;
    - any other non-empty list is sampled as a categorical choice among its
      elements;
    - any other value (scalar `int`/`float`/`str`/`bool`) is returned as is.

    Arguments
    ---------
    
    **trial** *(optuna.trial.Trial)*
    : The current Optuna trial, from which values are suggested.
    
    **value_name** *(str)*
    : The name of the hyperparameter (used as the parameter name in the trial
      and to exclude the `"evaluation"` key).
    
    **value** *(int or float or list)*
    : The value of the hyperparameter as written in the configuration (a
      scalar, a numeric range or a list of choices).
    
    Returns
    -------
    
    **suggested_value** *(int or float or list)*
    : The suggested value for this hyperparameter.
    
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