"""
Command line interface for KGATE.

Train and test a knowledge graph embedding model from the command line,
without writing any Python.

The CLI accepts an optional TOML configuration file (see
``kgate/config_template.toml`` for all available keys) plus one command line
option for every configuration key, with short, memorable aliases. Command
line options always take priority over the configuration file.

Typical use::

    # Train and test a model on your own knowledge graph (CSV with columns head, tail, edge)
    kgate my_kg.csv --output runs/my_run

    # Change the model, the embedding size and the number of epochs
    kgate my_kg.csv --output runs/my_run --decoder ComplEx --dim 50 --epochs 30

    # Start from a configuration file and override a few options
    kgate config.toml --lr 0.0005 --batch-size 4096

    # Run a built-in benchmark dataset
    kgate --dataset FB15k-237 --output runs/fb15k
"""

from __future__ import annotations

import argparse
import importlib.resources
import logging
import math
import re
import sys
import tomllib
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from kgate import Architect
from kgate.utils import find_best_model
from kgate.constants import (
    SUPPORTED_DECODERS,
    SUPPORTED_ENCODERS,
    SUPPORTED_INITIALIZERS,
    SUPPORTED_LOSSES,
    SUPPORTED_REGULARIZER_PARAMS,
    SUPPORTED_REGULARIZERS,
    SUPPORTED_SAMPLERS,
)

PROG = "kgate"

# --------------------------------------------------------------------------
# Value parsers (used as argparse `type=` functions)
# --------------------------------------------------------------------------


def _canonical_options(names: Sequence[str]) -> dict[str, str]:
    """Map case-insensitive, punctuation-free keys to canonical option names."""
    return {re.sub(r"[^a-z0-9]", "", name.lower()): name for name in names}


def _choice_type(options: dict[str, str], what: str) -> Callable[[str], str]:
    """Return an argparse type that maps user input to a canonical option name.

    Matching is case-insensitive and ignores punctuation, so e.g. ``transe``,
    ``TransE`` or ``trans-e`` all map to ``TransE``.
    """

    def convert(value: str) -> str:
        key = re.sub(r"[^a-z0-9]", "", value.lower())
        if key in options:
            return options[key]
        valid = ", ".join(sorted(set(options.values())))
        raise argparse.ArgumentTypeError(f"invalid {what} '{value}'. Choose from: {valid}")

    return convert


def _csv_list(value: str) -> list[str]:
    """Parse a comma-separated list of strings (e.g. ``E1,E2``)."""
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError(f"expected a comma-separated list (e.g. E1,E2), got '{value}'")
    return items


def _split_proportions(value: str) -> list[float]:
    """Parse the train/validation/test split (e.g. ``0.8,0.1,0.1``)."""
    try:
        parts = [float(item) for item in value.split(",") if item.strip() != ""]
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid split '{value}', expected e.g. 0.8,0.1,0.1") from None
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"the split must have exactly 3 values (train, validation, test), got {len(parts)}")
    if any(part < 0 for part in parts) or not math.isclose(sum(parts), 1.0, abs_tol=1e-6):
        raise argparse.ArgumentTypeError(f"the split proportions must be non-negative and sum to 1, got {value} (sum={sum(parts)})")
    return parts


def _key_value(value: str) -> tuple[str, Any]:
    """Parse a ``KEY=VALUE`` pair with basic type inference (e.g. ``gamma=0.9``)."""
    key, sep, raw = value.partition("=")
    key = key.strip()
    raw = raw.strip()
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"invalid parameter '{value}', expected KEY=VALUE (e.g. gamma=0.9)")
    lowered = raw.lower()
    if lowered in ("true", "false"):
        return key, lowered == "true"
    for cast in (int, float):
        try:
            return key, cast(raw)
        except ValueError:
            continue
    return key, raw


# --------------------------------------------------------------------------
# Precomputed option sets
# --------------------------------------------------------------------------

ENCODER_OPTIONS = _canonical_options(SUPPORTED_ENCODERS)
DECODER_OPTIONS = _canonical_options(SUPPORTED_DECODERS)
INITIALIZER_OPTIONS = _canonical_options(SUPPORTED_INITIALIZERS)
LOSS_OPTIONS = _canonical_options(SUPPORTED_LOSSES)
SAMPLER_OPTIONS = _canonical_options(SUPPORTED_SAMPLERS)
REGULARIZER_OPTIONS = _canonical_options(SUPPORTED_REGULARIZERS + ["None"])
REGULARIZER_PARAM_OPTIONS = _canonical_options(SUPPORTED_REGULARIZER_PARAMS)

OBJECTIVE_OPTIONS: dict[str, str] = {
    "linkprediction": "Link Prediction",
    "link": "Link Prediction",
    "lp": "Link Prediction",
    "tripletclassification": "Triplet Classification",
    "triplet": "Triplet Classification",
    "classification": "Triplet Classification",
    "tc": "Triplet Classification",
}

BUILTIN_DATASETS = ("FB15k-237", "WN18RR", "PrimeKG")


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    for name in ("KGATE", "kgate"):
        try:
            return version(name)
        except PackageNotFoundError:
            continue
    return "unknown"


# --------------------------------------------------------------------------
# Argument parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "KGATE — train and test a knowledge graph embedding model from the command line.\n"
            "\n"
            "A knowledge graph is a table of triplets (head, relation, tail), for example\n"
            '("Paris", "isCapitalOf", "France"). KGATE learns numerical representations of\n'
            "the nodes and relations so that missing links can be predicted.\n"
            "\n"
            "For most runs you only need to give it the knowledge graph (--kg-csv or --dataset)\n"
            "and an output directory (--output); everything else has sensible defaults.\n"
            "Every option below corresponds to a key of the configuration template, and\n"
            "command line options always override the configuration file."
        ),
        epilog=(
            "examples:\n"
            "  kgate my_kg.csv --output runs/my_run\n"
            "      Train and test a default model on a CSV file (columns: head, tail, edge).\n"
            "\n"
            "  kgate my_kg.csv --output runs/my_run --decoder ComplEx --dim 50 --epochs 30\n"
            "      Change the model, the embedding size and the number of epochs.\n"
            "\n"
            "  kgate config.toml --lr 0.0005 --batch-size 4096\n"
            "      Start from a configuration file, then override a few options.\n"
            "\n"
            "  kgate --dataset FB15k-237 --output runs/fb15k\n"
            "      Run the standard FB15k-237 benchmark (downloaded automatically).\n"
            "\n"
            "  kgate my_kg.csv --output runs/my_run --dry-run\n"
            "      Check that everything is set up correctly without training.\n"
        ),
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        metavar="config-or-kg",
        help=(
            "path to a TOML configuration file, or directly to the knowledge graph CSV "
            "file (in which case it is equivalent to --kg-csv). Command line options "
            "override the configuration file values."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_version()}")

    # --- Dataset & output -------------------------------------------------
    group = parser.add_argument_group("dataset & output")
    group.add_argument(
        "--kg-csv", "-k",
        metavar="PATH",
        help=(
            "path to a CSV file with the knowledge graph, one triplet per row, with the "
            "columns 'head', 'tail' and 'edge'. Extra columns are ignored."
        ),
    )
    group.add_argument(
        "--dataset", "-d",
        metavar="NAME",
        choices=list(BUILTIN_DATASETS),
        help=(
            "use a built-in benchmark dataset instead of a CSV file; the dataset is "
            "downloaded automatically. Choose from: " + ", ".join(BUILTIN_DATASETS)
        ),
    )
    group.add_argument(
        "--kg-pkl",
        metavar="PATH",
        help=(
            "path to a preprocessed knowledge graph pickle file (advanced). If both a "
            "CSV and a pickle are set, the pickle takes priority."
        ),
    )
    group.add_argument(
        "--metadata", "-m",
        metavar="PATH",
        help=(
            "path to a CSV of node metadata with the columns 'id' and 'type' (one row "
            "per node). Optional: without it, all nodes are considered of the same type."
        ),
    )
    group.add_argument(
        "--output", "-o",
        metavar="DIR",
        help=(
            "directory where checkpoints, metrics and logs are written. Required, either "
            "here or in the configuration file."
        ),
    )
    group.add_argument(
        "--seed",
        type=int,
        metavar="INT",
        help="random seed for reproducible runs (default: 42)",
    )

    # --- Model ------------------------------------------------------------
    group = parser.add_argument_group("model")
    group.add_argument(
        "--dim",
        type=int,
        metavar="INT",
        help=(
            "embedding dimension: the size of the vector learned for every node and "
            "relation. Bigger is more expressive but needs more memory and data "
            "(default: 256)"
        ),
    )
    group.add_argument(
        "--edge-dim",
        type=int,
        metavar="INT",
        help="embedding dimension for relations (default: same as --dim; use -1 to keep the default)",
    )
    group.add_argument(
        "--decoder",
        type=_choice_type(DECODER_OPTIONS, "decoder"),
        metavar="NAME",
        help=(
            "scoring function used to predict links. Common choices: TransE, DistMult, "
            "ComplEx. Choose from: " + ", ".join(DECODER_OPTIONS.values())
        ),
    )
    group.add_argument(
        "--dissimilarity",
        type=_choice_type({"l1": "L1", "l2": "L2", "torusl1": "torus_L1", "torusl2": "torus_L2", "torusel2": "torus_eL2"}, "dissimilarity"),
        metavar="NAME",
        help="distance used by TransE-like decoders: L1 or L2 (default: L2); TorusE also accepts torus_L1, torus_L2, torus_eL2",
    )
    group.add_argument(
        "--filters",
        type=int,
        metavar="INT",
        help="number of convolution filters for the ConvKB decoder (default: 3)",
    )
    group.add_argument(
        "--encoder",
        type=_choice_type(ENCODER_OPTIONS, "encoder"),
        metavar="NAME",
        help=(
            "graph neural network that aggregates local context before scoring: "
            "None (no encoder, default), GCN or GAT. Choose from: " + ", ".join(ENCODER_OPTIONS.values())
        ),
    )
    group.add_argument(
        "--gnn-layers",
        type=int,
        metavar="INT",
        help="number of GNN layers for GCN/GAT encoders (default: 1)",
    )
    group.add_argument(
        "--initializer",
        type=_choice_type(INITIALIZER_OPTIONS, "initializer"),
        metavar="NAME",
        help=(
            "how the initial embeddings are generated: Random (default), Feature or "
            "Node2Vec. Choose from: " + ", ".join(INITIALIZER_OPTIONS.values())
        ),
    )
    group.add_argument(
        "--walk-length",
        type=int,
        metavar="INT",
        help="Node2Vec only: number of steps per random walk (default: 10)",
    )
    group.add_argument(
        "--context-size",
        type=int,
        metavar="INT",
        help="Node2Vec only: context window size (default: 8)",
    )
    group.add_argument(
        "--regularizer",
        type=_choice_type(REGULARIZER_OPTIONS, "regularizer"),
        metavar="NAME",
        help=(
            "parameter normalization applied after each epoch: None (default), L1 or L2. "
            "Choose from: " + ", ".join(REGULARIZER_OPTIONS.values())
        ),
    )
    group.add_argument(
        "--regularize",
        type=_choice_type(REGULARIZER_PARAM_OPTIONS, "parameter selection"),
        metavar="NAME",
        help="which embeddings the regularizer is applied to: node (default), edge or all",
    )
    group.add_argument(
        "--loss",
        type=_choice_type(LOSS_OPTIONS, "loss"),
        metavar="NAME",
        help="training objective: Margin (default) or BCE. Choose from: " + ", ".join(LOSS_OPTIONS.values()),
    )
    group.add_argument(
        "--margin",
        type=int,
        metavar="INT",
        help="value by which positive triplets must outscore negative ones, for the Margin loss (default: 1)",
    )
    group.add_argument(
        "--reduction",
        type=_choice_type({"mean": "mean", "sum": "sum"}, "reduction"),
        metavar="NAME",
        help="how per-triplet losses are aggregated: mean (default) or sum",
    )

    # --- Training ----------------------------------------------------------
    group = parser.add_argument_group("training")
    group.add_argument(
        "--sampler", "-s",
        type=_choice_type(SAMPLER_OPTIONS, "negative sampler"),
        metavar="NAME",
        help=(
            "strategy to generate negative (fake) triplets during training. Choose from: "
            + ", ".join(SAMPLER_OPTIONS.values())
        ),
    )
    group.add_argument(
        "--negatives", "-n",
        type=int,
        metavar="INT",
        help="number of negative triplets generated per real triplet (default: 1)",
    )
    group.add_argument(
        "--optimizer",
        metavar="NAME",
        help="optimization algorithm, e.g. Adam (default), SGD or RMSprop (any PyTorch optimizer)",
    )
    group.add_argument(
        "--lr",
        type=float,
        metavar="FLOAT",
        help="learning rate (default: 0.001)",
    )
    group.add_argument(
        "--weight-decay",
        type=float,
        metavar="FLOAT",
        help="weight decay, a small L2 penalty that keeps the weights in check (default: 0.001)",
    )
    group.add_argument(
        "--scheduler",
        metavar="NAME",
        help=(
            "optional learning rate scheduler, e.g. StepLR, MultiStepLR, ExponentialLR, "
            "CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR (any PyTorch scheduler)"
        ),
    )
    group.add_argument(
        "--lr-param",
        type=_key_value,
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help=(
            "parameter for the learning rate scheduler, e.g. gamma=0.9 or milestones=30,60. "
            "Can be repeated; only used with --scheduler."
        ),
    )
    group.add_argument(
        "--epochs",
        type=int,
        metavar="INT",
        help="maximum number of training passes over the data (default: 100)",
    )
    group.add_argument(
        "--patience",
        type=int,
        metavar="INT",
        help="stop early if validation does not improve for this many evaluations (default: 20)",
    )
    group.add_argument(
        "--batch-size", "-b",
        type=int,
        metavar="INT",
        help="training batch size (default: 2048)",
    )
    group.add_argument(
        "--eval-batch-size",
        type=int,
        metavar="INT",
        help="batch size used for evaluation; keep it smaller than the training batch size (default: 32)",
    )
    group.add_argument(
        "--eval-every",
        type=int,
        metavar="INT",
        help="evaluate on the validation set every N epochs (default: 10)",
    )
    group.add_argument(
        "--save-every",
        type=int,
        metavar="INT",
        help="save a checkpoint every N epochs (default: 5)",
    )
    group.add_argument(
        "--keep-checkpoints",
        type=int,
        metavar="INT",
        help="number of checkpoints to keep on disk; -1 keeps all, 0 disables saving (default: 2)",
    )
    group.add_argument(
        "--pretrained",
        metavar="PATH|auto",
        help="path to a pretrained embedding file, or 'auto' to reuse the latest one in the output directory (default: auto)",
    )

    # --- Preprocessing -------------------------------------------------------
    group = parser.add_argument_group("preprocessing & data splitting")
    group.add_argument(
        "--preprocess",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="clean the KG and split it into train/validation/test before training (recommended for new KGs; default: on)",
    )
    group.add_argument(
        "--dedupe",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="remove duplicate triplets (default: on)",
    )
    group.add_argument(
        "--directed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="turn undirected edges into pairs of directed edges (default: on)",
    )
    group.add_argument(
        "--directed-edges",
        type=_csv_list,
        metavar="EDGES",
        help="comma-separated list of edge types to make directed (default: none, i.e. all)",
    )
    group.add_argument(
        "--flag-duplicates",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="flag near-duplicate edge types that could leak information between splits (default: on)",
    )
    group.add_argument(
        "--flag-cartesian",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="flag edge types forming cartesian products (default: on)",
    )
    group.add_argument(
        "--clean-train",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="remove flagged edges from the training set (default: on)",
    )
    group.add_argument(
        "--split",
        type=_split_proportions,
        metavar="TRAIN,VAL,TEST",
        help="proportions of train/validation/test sets; must sum to 1 (default: 0.8,0.1,0.1)",
    )
    group.add_argument(
        "--theta-cartesian",
        type=float,
        metavar="FLOAT",
        help="threshold (0 to 1) for the cartesian product check (default: 0.8)",
    )
    group.add_argument(
        "--theta-first",
        type=float,
        metavar="FLOAT",
        help="threshold (0 to 1) for the near-duplicate edge type check, first edge (default: 0.8)",
    )
    group.add_argument(
        "--theta-second",
        type=float,
        metavar="FLOAT",
        help="threshold (0 to 1) for the near-duplicate edge type check, second edge (default: 0.8)",
    )
    group.add_argument(
        "--permuted-edges",
        type=_csv_list,
        metavar="EDGES",
        help="comma-separated list of edge types to permute for the data leakage analysis (advanced; default: none)",
    )

    # --- Evaluation ----------------------------------------------------------
    group = parser.add_argument_group("evaluation")
    group.add_argument(
        "--objective",
        type=_choice_type(OBJECTIVE_OPTIONS, "evaluation objective"),
        metavar="NAME",
        help=(
            "what to measure on the test set: link prediction (default) or triplet "
            "classification (accepts 'link', 'lp', 'triplet', 'classification' or the full names)"
        ),
    )
    group.add_argument(
        "--target-edges",
        type=_csv_list,
        metavar="EDGES",
        help="comma-separated list of edge types to focus the evaluation on (default: all edge types)",
    )

    # --- Runtime --------------------------------------------------------------
    group = parser.add_argument_group("runtime")
    group.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="print the full training log (default: on; use --no-verbose for a quiet run)",
    )
    group.add_argument(
        "--cudnn-benchmark",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="let PyTorch benchmark convolution algorithms on GPU (default: on)",
    )
    group.add_argument(
        "--cores",
        type=int,
        metavar="INT",
        help="number of CPU threads to use (default: all available)",
    )
    group.add_argument(
        "--checkpoint",
        metavar="PATH",
        help="resume training from this checkpoint file instead of starting from scratch (advanced)",
    )
    group.add_argument(
        "--dry-run",
        action="store_true",
        help="initialize everything and validate the setup, but do not train or test",
    )

    return parser


# --------------------------------------------------------------------------
# Configuration assembly
# --------------------------------------------------------------------------


def build_config_dict(args: argparse.Namespace) -> dict:
    """Build the nested inline configuration dictionary from the parsed arguments.

    Only the options the user actually gave are included, so that the
    configuration file and the KGATE defaults fill in the rest.
    """
    cfg: dict[str, Any] = {}

    def put(path: tuple[str, ...], key: str, value: Any) -> None:
        node = cfg
        for part in path:
            node = node.setdefault(part, {})
        node[key] = value

    # Top level
    if args.seed is not None:
        put((), "seed", args.seed)
    if args.kg_csv is not None:
        put((), "kg_csv", args.kg_csv)
    if args.kg_pkl is not None:
        put((), "kg_pkl", args.kg_pkl)
    if args.metadata is not None:
        put((), "metadata_csv", args.metadata)
    if args.output is not None:
        put((), "output_directory", args.output)
    if args.verbose is not None:
        put((), "verbose", args.verbose)

    # [preprocessing]
    pre: dict[str, Any] = {}
    if args.preprocess is not None:
        pre["run_preprocessing"] = args.preprocess
    if args.dedupe is not None:
        pre["remove_duplicate_triplets"] = args.dedupe
    if args.directed is not None:
        pre["make_directed"] = args.directed
    if args.directed_edges is not None:
        pre["make_directed_edges"] = args.directed_edges
    if args.flag_duplicates is not None:
        pre["flag_near_duplicate_edges"] = args.flag_duplicates
    if args.flag_cartesian is not None:
        pre["flag_cartesian_edges"] = args.flag_cartesian
    if args.clean_train is not None:
        pre["clean_train_set"] = args.clean_train
    if args.split is not None:
        pre["split"] = args.split
    if args.theta_cartesian is not None:
        pre["theta_cartesian"] = args.theta_cartesian
    if args.theta_first is not None:
        pre["theta_first_edge_type"] = args.theta_first
    if args.theta_second is not None:
        pre["theta_second_edge_type"] = args.theta_second
    if pre:
        cfg["preprocessing"] = pre

    # [model] and its subsections
    model: dict[str, Any] = {}
    if args.dim is not None:
        model["node_embedding_dimensions"] = args.dim
    if args.edge_dim is not None:
        model["edge_embedding_dimensions"] = args.edge_dim

    init: dict[str, Any] = {}
    if args.initializer is not None:
        init["name"] = args.initializer
    if args.walk_length is not None:
        init["walk_length"] = args.walk_length
    if args.context_size is not None:
        init["context_size"] = args.context_size
    if init:
        model["initializer"] = init

    enc: dict[str, Any] = {}
    if args.encoder is not None:
        enc["name"] = args.encoder
    if args.gnn_layers is not None:
        enc["gnn_layer_number"] = args.gnn_layers
    if enc:
        model["encoder"] = enc

    dec: dict[str, Any] = {}
    if args.decoder is not None:
        dec["name"] = args.decoder
    if args.dissimilarity is not None:
        dec["dissimilarity"] = args.dissimilarity
    if args.filters is not None:
        dec["filter_count"] = args.filters
    if dec:
        model["decoder"] = dec

    reg: dict[str, Any] = {}
    if args.regularizer is not None:
        reg["name"] = args.regularizer
    if args.regularize is not None:
        reg["params"] = args.regularize
    if reg:
        model["regularizer"] = reg

    loss: dict[str, Any] = {}
    if args.loss is not None:
        loss["name"] = args.loss
    if args.margin is not None:
        loss["margin"] = args.margin
    if args.reduction is not None:
        loss["reduction"] = args.reduction
    if loss:
        model["loss"] = loss

    if model:
        cfg["model"] = model

    # [negative_sampler]
    sampler: dict[str, Any] = {}
    if args.sampler is not None:
        sampler["name"] = args.sampler
    if args.negatives is not None:
        sampler["negative_triplet_count"] = args.negatives
    if sampler:
        cfg["negative_sampler"] = sampler

    # [optimizer]
    opt: dict[str, Any] = {}
    if args.optimizer is not None:
        opt["name"] = args.optimizer
    opt_params: dict[str, Any] = {}
    if args.lr is not None:
        opt_params["learning_rate"] = args.lr
    if args.weight_decay is not None:
        opt_params["weight_decay"] = args.weight_decay
    if opt_params:
        opt["params"] = opt_params
    if opt:
        cfg["optimizer"] = opt

    # [learning_rate_scheduler]
    sched: dict[str, Any] = {}
    if args.scheduler is not None:
        sched["name"] = args.scheduler
    if args.lr_param:
        params = sched.setdefault("params", {})
        for key, value in args.lr_param:
            params[key] = value
    if sched:
        cfg["learning_rate_scheduler"] = sched

    # [training]
    train: dict[str, Any] = {}
    if args.epochs is not None:
        train["max_epochs"] = args.epochs
    if args.patience is not None:
        train["patience"] = args.patience
    if args.batch_size is not None:
        train["train_batch_size"] = args.batch_size
    if args.eval_batch_size is not None:
        train["evaluation_batch_size"] = args.eval_batch_size
    if args.eval_every is not None:
        train["evaluation_interval"] = args.eval_every
    if args.save_every is not None:
        train["save_interval"] = args.save_every
    if args.keep_checkpoints is not None:
        train["keep_n_checkpoints"] = args.keep_checkpoints
    if args.pretrained is not None:
        train["pretrained_embeddings"] = args.pretrained
    if train:
        cfg["training"] = train

    # [evaluation]
    evaluation: dict[str, Any] = {}
    if args.objective is not None:
        evaluation["objective"] = args.objective
    if args.target_edges is not None:
        evaluation["target_edges"] = args.target_edges
    if evaluation:
        cfg["evaluation"] = evaluation

    # [data_leakage]
    if args.permuted_edges is not None:
        cfg["data_leakage"] = {"permuted_edges": args.permuted_edges}

    return cfg


def load_default_template() -> dict:
    """Load the shipped default configuration (config_template.toml)."""
    with importlib.resources.open_binary("kgate", "config_template.toml") as f:
        return tomllib.load(f)


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge two dictionaries; values from `override` win."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


# --------------------------------------------------------------------------
# Validation, summary and reporting
# --------------------------------------------------------------------------


def validate(args: argparse.Namespace, file_cfg: dict) -> tuple[list[str], list[str]]:
    """Check the command line arguments and return (errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []

    # Required: an output directory
    out_dir = args.output or file_cfg.get("output_directory") or ""
    if not out_dir:
        errors.append(
            "no output directory given. Use --output DIR (or set 'output_directory' in the configuration file) "
            "so that KGATE knows where to write checkpoints and metrics."
        )

    # Required: a knowledge graph, in any form
    if not (args.dataset or args.kg_csv or args.kg_pkl or file_cfg.get("kg_csv") or file_cfg.get("kg_pkl")):
        errors.append(
            "no knowledge graph given. Provide one with --kg-csv PATH, --dataset NAME or --kg-pkl PATH "
            "(or 'kg_csv'/'kg_pkl' in the configuration file)."
        )

    # Numeric ranges (friendly messages instead of deep assertion errors)
    range_checks = [
        (args.dim, 1, None, "--dim must be a positive integer"),
        (args.edge_dim, -1, None, "--edge-dim must be a positive integer or -1 (for the same size as --dim)"),
        (args.negatives, 1, None, "--negatives must be at least 1"),
        (args.epochs, 1, None, "--epochs must be at least 1"),
        (args.patience, 1, None, "--patience must be at least 1"),
        (args.batch_size, 1, None, "--batch-size must be at least 1"),
        (args.eval_batch_size, 1, None, "--eval-batch-size must be at least 1"),
        (args.eval_every, 1, None, "--eval-every must be at least 1"),
        (args.save_every, 1, None, "--save-every must be at least 1"),
        (args.keep_checkpoints, -1, None, "--keep-checkpoints must be -1 (keep all) or a non-negative integer"),
        (args.gnn_layers, 0, None, "--gnn-layers must be 0 or a positive integer"),
        (args.walk_length, 1, None, "--walk-length must be at least 1"),
        (args.context_size, 1, None, "--context-size must be at least 1"),
        (args.margin, 0, None, "--margin must be a non-negative number"),
        (args.theta_cartesian, 0, 1, "--theta-cartesian must be between 0 and 1"),
        (args.theta_first, 0, 1, "--theta-first must be between 0 and 1"),
        (args.theta_second, 0, 1, "--theta-second must be between 0 and 1"),
        (args.lr, None, None, "--lr must be a positive number"),
        (args.weight_decay, 0, 1, "--weight-decay must be between 0 and 1"),
    ]
    for value, low, high, message in range_checks:
        if value is None:
            continue
        if low is not None and value < low:
            errors.append(message)
        elif high is not None and value > high:
            errors.append(message)
        elif message.startswith("--lr ") and value <= 0:
            errors.append(message)

    # Scheduler / optimizer sanity checks against PyTorch
    if args.scheduler is not None:
        import torch.optim.lr_scheduler as schedulers
        if not hasattr(schedulers, args.scheduler):
            errors.append(
                f"'{args.scheduler}' is not a PyTorch learning rate scheduler. "
                "See https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate for the available ones."
            )
    if args.optimizer is not None:
        import torch.optim as optim
        if not hasattr(optim, args.optimizer):
            errors.append(f"'{args.optimizer}' is not a PyTorch optimizer. Common choices: Adam, SGD, RMSprop.")
    if args.lr_param:
        effective_scheduler = args.scheduler or (file_cfg.get("learning_rate_scheduler") or {}).get("name", "")
        if not effective_scheduler:
            warnings.append("--lr-param is set but no learning rate scheduler is; the parameters will be ignored. Use --scheduler to activate one.")

    if args.checkpoint is not None and not Path(args.checkpoint).is_file():
        errors.append(f"--checkpoint file not found: {args.checkpoint}")

    return errors, warnings


def print_summary(effective: dict, args: argparse.Namespace) -> None:
    """Print a plain-language summary of the resolved run configuration."""
    model = effective.get("model", {})
    loss = model.get("loss", {})
    decoder = model.get("decoder", {})
    encoder = model.get("encoder", {})
    initializer = model.get("initializer", {})
    sampler = effective.get("negative_sampler", {})
    optimizer = effective.get("optimizer", {})
    optimizer_params = optimizer.get("params", {})
    training = effective.get("training", {})
    preprocessing = effective.get("preprocessing", {})
    evaluation = effective.get("evaluation", {})

    if args.dataset:
        source = f"built-in dataset '{args.dataset}'"
    elif args.kg_csv:
        source = f"CSV file: {args.kg_csv}"
    elif args.kg_pkl:
        source = f"pickle file: {args.kg_pkl}"
    else:
        source = "from the configuration file"

    edge_dim = model.get("edge_embedding_dimensions")
    if edge_dim == -1:
        edge_dim = f"{model.get('node_embedding_dimensions')} (same as nodes)"

    lines = [
        "",
        f"KGATE {_version()} — knowledge graph embedding",
        f"  Knowledge graph   : {source}",
        f"  Output directory  : {effective.get('output_directory', '')}",
        f"  Decoder           : {decoder.get('name', '')} (dissimilarity: {decoder.get('dissimilarity', '')})",
        f"  Encoder           : {encoder.get('name', '')}",
        f"  Initializer       : {initializer.get('name', '')}",
        f"  Dimensions        : nodes {model.get('node_embedding_dimensions')}, edges {edge_dim}",
        f"  Loss              : {loss.get('name', '')} (margin: {loss.get('margin', '')}, reduction: {loss.get('reduction', '')})",
        f"  Negative sampler  : {sampler.get('name', '')} ({sampler.get('negative_triplet_count', '')} negative(s) per triplet)",
        f"  Optimizer         : {optimizer.get('name', '')} (lr: {optimizer_params.get('learning_rate', '')}, weight decay: {optimizer_params.get('weight_decay', '')})",
        f"  Training          : up to {training.get('max_epochs', '')} epochs, early stopping patience {training.get('patience', '')}",
        f"  Data split        : {' / '.join(str(part) for part in preprocessing.get('split', []))}",
        f"  Evaluation        : {evaluation.get('objective', '')}",
        f"  Seed              : {effective.get('seed', '')}",
        f"  Device            : {'cuda' if torch.cuda.is_available() else 'cpu'}",
        "",
    ]
    print("\n".join(lines))


def _format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_results(results: dict) -> None:
    """Pretty-print the test results returned by Architect.test()."""
    print("=== Test results (best model) ===")
    if "Global_metrics" in results:
        print(f"  Overall metric: {_format_metric(results['Global_metrics'])}")

    def render(label: str, block: Any) -> None:
        if not isinstance(block, dict):
            return
        if "Global_metrics" in block:
            print(f"  {label}: {_format_metric(block['Global_metrics'])}")
        individual = block.get("Individual_metrics")
        if isinstance(individual, dict) and individual:
            for edge, metric in individual.items():
                print(f"    {edge}: {_format_metric(metric)}")

    if "target_edges" in results:
        render("Target edges", results["target_edges"])
    if "remaining_edges" in results:
        render("Remaining edges", results["remaining_edges"])


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def _looks_like_kg_csv(path: Path) -> bool:
    """Heuristic: does this file look like a knowledge graph CSV?"""
    if path.suffix.lower() == ".csv":
        return True
    try:
        with open(path, "r", errors="replace") as f:
            header = f.readline().lower()
    except OSError:
        return False
    return "head" in header and ("tail" in header or "edge" in header)


def resolve_positional(args: argparse.Namespace) -> tuple[str | None, list[str], int | None]:
    """Interpret the positional argument as a config file or a KG CSV file.

    Returns (config_path, warnings, exit_code). exit_code is None on success.
    If the positional is a KG CSV, args.kg_csv is updated accordingly.
    """
    if args.config is None:
        return None, [], None

    candidate = Path(args.config)
    if candidate.is_file():
        try:
            with open(candidate, "rb") as f:
                tomllib.load(f)
            return args.config, [], None  # Valid TOML: it is a configuration file.
        except tomllib.TOMLDecodeError:
            pass

    # Not valid TOML: the most likely intent is that the user passed the KG CSV directly.
    if candidate.is_file() and _looks_like_kg_csv(candidate):
        if args.kg_csv is not None:
            return None, [], 2  # Ambiguous: the caller will report it.
        warnings = [f"'{args.config}' is not a TOML configuration file; treating it as the knowledge graph CSV (same as --kg-csv)."]
        args.kg_csv = args.config
        return None, warnings, None

    return None, [], 2


def run(args: argparse.Namespace, cli_cfg: dict, config_path: str | None) -> int:
    """Instantiate the Architect, train the model and run the test evaluation."""
    extra: dict[str, Any] = {}
    if args.dataset:
        extra["knowledge_graph"] = args.dataset
    if args.cudnn_benchmark is not None:
        extra["cudnn_benchmark"] = args.cudnn_benchmark
    if args.cores is not None:
        extra["number_of_cores"] = args.cores

    print("Starting the run (this initializes the model and trains it)...")
    architect = Architect(
        config_path=config_path or "",
        **cli_cfg,
        **extra,
    )

    architect.train_model(
        checkpoint_file=Path(args.checkpoint) if args.checkpoint else None,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("\nDry run completed: the setup is valid and the model was initialized. No training or test was performed.")
        return 0

    test_count = int(architect.knowledge_graph.test_mask.sum().item())
    if test_count == 0:
        print("\nTraining completed, but the test set is empty, so `test()` is skipped.")
        print("Check --split (the splitter keeps at least one training triplet per head-tail pair, so a test triplet requires at least three triplets per pair) or use a larger dataset.")
        return 0

    try:
        has_checkpoint = find_best_model(architect.checkpoints_directory) is not None
    except OSError:
        has_checkpoint = False
    if not has_checkpoint:
        print("\nNote: no checkpoint was saved during training (validation runs every "
              f"{architect.configuration.training.evaluation_interval} epoch(s), and training stopped before that). "
              "The model currently in memory will be evaluated directly.")

    print("\nTraining complete. Running the test evaluation...")
    results = architect.test()
    print()
    print_results(results)
    output_directory = architect.configuration.output_directory
    print(f"\nMetrics saved to {Path(output_directory) / 'evaluation_metrics.toml'}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Interpret the positional argument: config file, or directly the KG CSV.
    config_path, positional_warnings, exit_code = resolve_positional(args)
    if exit_code is not None:
        if args.kg_csv is not None:
            print(f"error: '{args.config}' looks like a knowledge graph CSV, but --kg-csv was already given. Remove one of them.", file=sys.stderr)
        else:
            print(f"error: '{args.config}' is neither a valid TOML configuration file nor a knowledge graph CSV.", file=sys.stderr)
            print("hint: to give the knowledge graph, use --kg-csv PATH; to give a configuration, pass a .toml file.", file=sys.stderr)
        return exit_code

    # Read the configuration file (if any) early, to merge it into checks and the summary.
    file_cfg: dict = {}
    if config_path is not None:
        try:
            with open(config_path, "rb") as f:
                file_cfg = tomllib.load(f)
        except tomllib.TOMLDecodeError as exc:
            print(f"error: could not parse {config_path} as TOML: {exc}", file=sys.stderr)
            return 2

    cli_cfg = build_config_dict(args)

    errors, warnings = validate(args, file_cfg)
    for warning in positional_warnings + warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        print("Run 'kgate --help' for usage and examples.", file=sys.stderr)
        return 2

    # Resolve the effective configuration (defaults < file < command line).
    try:
        template = load_default_template()
    except Exception:
        template = {}
    effective = deep_merge(deep_merge(template, file_cfg), cli_cfg)

    # Wire the 'verbose' setting to the logging level, so --no-verbose actually works.
    verbose = bool(effective.get("verbose", True))
    logging.getLogger().setLevel(logging.INFO if verbose else logging.WARNING)

    print_summary(effective, args)

    try:
        return run(args, cli_cfg, config_path)
    except KeyboardInterrupt:
        print("\nInterrupted. Checkpoints already saved in the output directory can be used to resume with --checkpoint.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        if verbose:
            import traceback

            traceback.print_exc()
        else:
            print("hint: re-run with --verbose to see the full traceback.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
