"""
Grid search / hyperparameter optimization for KGATE.

A hyperparameter is registered in the grid search by writing it as a list in
the configuration file (see `run_grid_search` for the exact list syntax).
`run_grid_search` then runs an Optuna study (``direction="maximize"``,
median pruner) that maximizes the ``"Global_metrics"`` entry of
`Architect.test()`.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

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


def _is_number(value: Any) -> bool:
    """True if value is an int or float (bools are excluded on purpose)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _range_spec(value: List[Any]) -> Optional[Tuple[float, float, Optional[float], bool]]:
    """
    Interpret a list as a numeric search range, if it is well-formed.

    Recognized forms (with low < high):
    - ``[low, high]``          -> uniform range, no step, linear scale
    - ``[low, high, step]``    -> uniform range with the given step (step > 0)
    - ``[low, high, True]``    -> uniform range on a logarithmic scale (low > 0)

    Returns
    -------
    
    **(low, high, step, log) tuple, or None**
    : None when the list is not a well-formed numeric range (e.g. a list of
      data values such as the ``preprocessing.split`` proportions, or a list
      of categorical choices).
    
    """
    if len(value) not in (2, 3):
        return None

    low, high = value[0], value[1]
    if not (_is_number(low) and _is_number(high)) or low >= high:
        return None

    step: Optional[float] = None
    log = False
    if len(value) == 3:
        third = value[2]
        if isinstance(third, bool):
            if not third:
                return None  # [low, high, False] is not a documented form
            log = True
        elif _is_number(third):
            if third <= 0:
                return None
            step = third
        else:
            return None

    if log and low <= 0:
        return None  # Optuna requires low > 0 for a logarithmic scale

    return low, high, step, log


def _collect_search_space(config: Dict[str, Any],
                          seen: Optional[Dict[str, Any]] = None
                          ) -> Dict[str, Any]:
    """
    Walk a (raw) configuration dictionary and collect the hyperparameters
    that will be sampled, keyed by the parameter name used in the trial.

    The parameter name is the configuration key name (e.g. ``"name"``,
    ``"learning_rate"``), which is exactly what `suggest_value` passes to
    ``trial.suggest_*``. A mapping is returned so that `run_grid_search` can
    detect name collisions before the study starts (Optuna does not allow the
    same parameter name to be suggested with two different value spaces in
    one trial).

    Arguments
    ---------
    
    **config** *(dict)*
    : The raw (merged) configuration dictionary.
    
    **seen** *(dict, optional)*
    : Accumulator mapping parameter name -> value space spec. A fresh empty
      dictionary is used when not given.
    
    Returns
    -------
    
    **seen** *(dict)*
    : The mapping of sampled parameter name -> value space spec.
    
    """
    if seen is None:
        seen = {}

    for key, value in config.items():
        if key == "evaluation":
            continue  # never sampled (see suggest_value)

        if isinstance(value, dict):
            _collect_search_space(value, seen)

        elif isinstance(value, list) and len(value) > 0:
            spec = _range_spec(value)
            if spec is not None:
                space = ("range", spec)
            elif any(not _is_number(element) for element in value):
                space = ("categorical", tuple(value))
            else:
                continue  # numeric list of data values (e.g. split proportions): not searched

            if key in seen and seen[key] != space:
                raise ValueError(
                    f"Grid search: the parameter name '{key}' is used for two "
                    f"different value spaces ({seen[key]} and {space}). Optuna "
                    f"identifies hyperparameters by their configuration key "
                    f"name, which must therefore be unique among the searched "
                    f"hyperparameters."
                )
            seen[key] = space

    return seen


def _attach_pruning(architect: Architect,
                    trial: optuna.trial.Trial) -> None:
    """
    Report the per-epoch validation metric to the Optuna trial and prune the
    trial when the median pruner deems it uncompetitive.

    `Architect.train_model` registers `Architect.evaluate` as an Ignite event
    handler at the moment it is called, so shadowing the bound method on the
    instance (before `train_model` is called) is enough to intercept every
    validation evaluation without modifying the Architect.

    Arguments
    ---------
    
    **architect** *(Architect)*
    : The Architect whose training run should be prunable.
    
    **trial** *(optuna.trial.Trial)*
    : The current Optuna trial.
    
    """
    original_evaluate = architect.evaluate

    def pruning_evaluate(engine) -> None:
        original_evaluate(engine)
        value = engine.state.metrics.get("validation_metric_value")
        if value is None:
            return
        trial.report(float(value), engine.state.epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    architect.evaluate = pruning_evaluate


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
    
    Returns
    -------
    
    **best_trial** *(optuna.trial.FrozenTrial or None)*
    : The best completed trial (its ``params`` hold the suggested
      hyperparameters), or None if every trial was pruned or failed.
    
    Notes
    -----
    
    - The configuration is parsed once, before the study starts; each trial
      then suggests a value for every registered list and trains one full
      Architect with those values.
    
    - A list is interpreted as a numeric range only when it is well-formed
      (``low < high``, ``step > 0`` when given, ``low > 0`` for the
      logarithmic form). Any other list containing at least one non-numeric
      element (e.g. ``["TransE", "ComplEx"]``) is a list of categorical
      choices. A purely numeric list that is not a valid range - such as the
      ``preprocessing.split`` proportions ``[0.8, 0.1, 0.1]`` - is treated as
      data and left unchanged. Empty lists and the whole ``evaluation``
      section are never sampled.
    
    - Optuna identifies hyperparameters by their configuration key name, so
      two searched hyperparameters must not share a key name with different
      value spaces (e.g. two different ``name`` choice lists); this is
      checked up front and raises a `ValueError` before any training starts.
    
    - The median pruner is active: the validation metric is reported to
      Optuna at every `Architect.evaluate` call (i.e. every
      ``evaluation_interval`` epochs) and uncompetitive trials are stopped
      early.
    
    If the configuration file has no hyperparameter list, this function is
    effectively the same as running `Architect(config_path).train_model()`
    (repeated `number_of_trials` times, with the same hyperparameters).
    
    """
    raw: Dict[str, Any] = Configuration.parse(config_path, {})

    search_space = _collect_search_space(raw)
    if search_space:
        logging.info(f"Grid search over {len(search_space)} hyperparameter(s):")
        for key, space in search_space.items():
            logging.info(f"  {key}: {space[0]} {space[1]}")
    else:
        logging.info("No hyperparameter list found in the configuration: "
                     "running the same configuration for every trial.")

    def objective(trial: optuna.trial.Trial) -> float:
        resolved: Dict[str, Any] = {key: suggest_value(trial, key, raw[key]) for key in raw}

        architect = Architect(  config_path = config_path,
                                knowledge_graph = kg,
                                dataframe = dataframe,
                                **resolved)

        _attach_pruning(architect, trial)

        architect.train_model()

        result = architect.test()
        
        return float(result["Global_metrics"])

    study = optuna.create_study(direction = "maximize",
                                pruner = optuna.pruners.MedianPruner())
    study.optimize( objective,
                    n_trials = number_of_trials)

    best_trial = study.best_trial
    if best_trial is None:
        logging.warning("Grid search finished without any completed trial "
                        "(all trials were pruned or failed).")
        return None

    logging.info(f"Best trial score: {best_trial.value}")
    logging.info("Best trial hyperparameters:")
    for key, value in best_trial.params.items():
        logging.info(f"  {key}: {value}")

    return best_trial


def suggest_value(  trial: optuna.trial.Trial,
                    value_name: str,
                    value: Any,
                    ) -> Any:
    """
    Suggest a value for a single hyperparameter from an Optuna trial, based on
    how it is written in the configuration.

    Rules applied:
    - the `"evaluation"` key is never sampled (returned as is);
    - a `dict` is recursed into, key by key;
    - an empty list is returned as is;
    - a well-formed list of the form `[low, high]`, `[low, high, step]`
      (low < high, step > 0) or `[low, high, True]` (low > 0) is sampled with
      `trial.suggest_float` or `trial.suggest_int` over `[low, high]` with the
      given step, or on a logarithmic scale when the third element is `True`;
      a float is suggested as soon as any of the range bounds or the step is a
      float, otherwise an int;
    - any other non-empty list containing at least one non-numeric element is
      sampled as a categorical choice among its elements;
    - a purely numeric list that is not a well-formed range (e.g. the
      `preprocessing.split` proportions) is treated as data and returned as is;
    - any other value (scalar `int`/`float`/`str`/`bool`) is returned as is.

    Arguments
    ---------
    
    **trial** *(optuna.trial.Trial)*
    : The current Optuna trial, from which values are suggested.
    
    **value_name** *(str)*
    : The name of the hyperparameter (used as the parameter name in the trial
      and to exclude the `"evaluation"` key).
    
    **value** *(Any)*
    : The value of the hyperparameter as written in the configuration (a
      scalar, a numeric range, a list of choices or a nested configuration
      section).
    
    Returns
    -------
    
    **suggested_value** *(Any)*
    : The suggested value for this hyperparameter (a scalar for searched
      parameters, the original value otherwise).
    
    """
    if value_name == "evaluation":
        return value

    if isinstance(value, dict):
        return {child_key: suggest_value(trial, child_key, value[child_key])
                for child_key in value}

    if isinstance(value, list):
        if len(value) == 0:
            return value

        spec = _range_spec(value)
        if spec is not None:
            low, high, step, log = spec
            if any(isinstance(element, float) for element in (low, high, step)
                   if element is not None):
                return trial.suggest_float(  name = value_name,
                                             low = low,
                                             high = high,
                                             step = step,
                                             log = log)
            if step is None:
                # IntDistribution requires a step (default 1); step=None is
                # only valid for float distributions.
                return trial.suggest_int(    name = value_name,
                                             low = low,
                                             high = high,
                                             log = log)
            return trial.suggest_int(    name = value_name,
                                         low = low,
                                         high = high,
                                         step = step,
                                         log = log)

        if any(not _is_number(element) for element in value):
            return trial.suggest_categorical(name = value_name, choices = value)

        return value  # numeric data list (e.g. split proportions): keep as is

    return value
