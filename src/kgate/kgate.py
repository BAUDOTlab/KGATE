import os
from inspect import signature
from pathlib import Path
from torchkge.models import Model
import torchkge.sampling as sampling
import torch
from torch import tensor, long, stack
from torch.nn.functional import normalize
from .utils import parse_config, load_knowledge_graph, set_random_seeds, find_best_model, create_hetero_data, init_embedding
from .preprocessing import prepare_knowledge_graph
from .encoders import *
from .decoders import *
from .data_structures import KGATEGraph
from .samplers import FixedPositionalNegativeSampler
from .evaluators import KLinkPredictionEvaluator
from torchkge.utils import MarginLoss, BinaryCrossEntropyLoss, DataLoader
import logging
import warnings
import torch.optim as optim
from torch.optim import lr_scheduler
import csv
import gc
from ignite.metrics import RunningAverage
from ignite.engine import Events, Engine
from ignite.handlers import EarlyStopping, ModelCheckpoint, Checkpoint, DiskSaver
import pandas as pd
import yaml

# Configure logging
logging.captureWarnings(True)
log_level = logging.INFO# if config["common"]['verbose'] else logging.WARNING
logging.basicConfig(
    level=log_level,  
    format='%(asctime)s - %(levelname)s - %(message)s' 
)

TRANSLATIONAL_MODELS = ['TransE', 'TransH', 'TransR', 'TransD', 'TorusE']

class Architect(Model):
    def __init__(self, kg = None, config_path: str = "", cudnn_benchmark = True, num_cores = 0, **kwargs):
        # kg should be of type KGATEGraph or KnowledgeGraph, if exists use it instead of the one in config
        self.config = parse_config(config_path, kwargs)

        if torch.cuda.is_available():
            # Benchmark convolution algorithms to chose the optimal one.
            # Initialization is slightly longer when it is enabled.
            torch.backends.cudnn.benchmark = cudnn_benchmark

        # If given, restrict the parallelisation to user-defined threads.
        # Otherwise, use all the cores the process has access to.
        num_cores = num_cores if num_cores > 0 else len(os.sched_getaffinity(0))
        logging.info(f"Setting number of threads to {num_cores}")
        torch.set_num_threads(num_cores)

        outdir = Path(self.config["output_directory"])
        # Create output folder if it doesn't exist
        logging.info(f"Output folder: {outdir}")
        outdir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = outdir.joinpath("checkpoints")


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Detected device: {self.device}')

        set_random_seeds(self.config["seed"])

        self.emb_dim = self.config["model"]["emb_dim"]
        self.rel_emb_dim = self.config["model"]["rel_emb_dim"]

        run_kg_prep =  self.config["run_kg_preprocess"]

        if run_kg_prep:
            logging.info(f"Preparing KG...")
            self.kg_train, self.kg_val, self.kg_test = prepare_knowledge_graph(self.config)
            logging.info("KG preprocessed.")
        else:
            logging.info("Loading KG...")
            self.kg_train, self.kg_val, self.kg_test = load_knowledge_graph(self.config["kg_pkl"])
            logging.info("Done")

        super().__init__(self.kg_train.n_ent, self.kg_train.n_rel)


    def initialize_encoder(self):
        metadata_csv = self.config["metadata_csv"]
        encoder_config = self.config["model"]["encoder"]
        encoder_name = encoder_config["name"]
        gnn_layers = encoder_config["gnn_layer_number"]

        match encoder_name:
            case "Default":
                encoder = DefaultEncoder()
            case "GCN": 
                encoder = GCNEncoder(self.node_embeddings, self.hetero_data, self.emb_dim, gnn_layers)
            case "GAT":
                encoder = GATEncoder(self.node_embeddings, self.hetero_data, self.emb_dim, gnn_layers)

        return encoder

    def initialize_decoder(self):
        decoder_config = self.config["model"]["decoder"]
        decoder_name = decoder_config["name"]
        dissimilarity = decoder_config["dissimilarity"]
        margin = decoder_config["margin"]

        # Translational models
        match decoder_name:
            case "TransE":
                decoder = TransE(self.emb_dim, self.kg_train.n_ent, self.kg_train.n_rel,
                               dissimilarity_type=dissimilarity)
                criterion = MarginLoss(margin)
            case "RESCAL":
                decoder = RESCAL(self.emb_dim, self.kg_train.n_ent, self.kg_train.n_rel)
                criterion = BinaryCrossEntropyLoss()
            case "DistMult":
                decoder = DistMult(self.emb_dim, self.kg_train.n_ent, self.kg_train.n_rel)
                criterion = BinaryCrossEntropyLoss()

        return decoder, criterion

    def initialize_optimizer(self):
        """
        Initialize the optimizer based on the configuration provided.
        
        Returns:
        - optimizer: Initialized optimizer.
        """

        optimizer_name = self.config["optimizer"]["name"]

        # Retrieve optimizer parameters, defaulting to an empty dict if not specified
        optimizer_params = self.config["optimizer"]["params"]

        # Mapping of optimizer names to their corresponding PyTorch classes
        optimizer_mapping = {
            'Adam': optim.Adam,
            'SGD': optim.SGD,
            'RMSprop': optim.RMSprop,
            # Add other optimizers as needed
        }

        # Check if the specified optimizer is supported
        if optimizer_name not in optimizer_mapping:
            raise ValueError(f"Optimizer type '{optimizer_name}' is not supported. Please check the configuration. Supported optimizers are :\n{"\n".join(optimizer_mapping.keys())}")

        optimizer_class = optimizer_mapping[optimizer_name]
        
        try:
            # Initialize the optimizer with given parameters
            optimizer = optimizer_class(self.decoder.parameters(), **optimizer_params)
        except TypeError as e:
            raise ValueError(f"Error initializing optimizer '{optimizer_name}': {e}")
        
        logging.info(f"Optimizer '{optimizer_name}' initialized with parameters: {optimizer_params}")
        return optimizer

    def initialize_sampler(self):
        """Initialize the sampler according to the configuration.
        
            Returns:
            - sampler: the initialized sampler"""
        
        sampler_config = self.config["sampler"]
        sampler_name = sampler_config["name"]
        n_neg = sampler_config["n_neg"]

        match sampler_name:
            case "Positional":
                sampler = FixedPositionalNegativeSampler(self.kg_train, self.kg_val, self.kg_test)
            case "Uniform":
                sampler = sampling.UniformNegativeSampler(self.kg_train, self.kg_val, self.kg_test, n_neg)
            case "Bernoulli":
                sampler = sampling.BernoulliNegativeSampler(self.kg_train, self.kg_val, self.kg_test, n_neg)
            case _:
                raise ValueError(f"Sampler type '{sampler_name}' is not supported. Please check the configuration.")
            
        return sampler
    
    def initialize_scheduler(self):
        """
        Initializes the learning rate scheduler based on the provided configuration.
                
        Returns:
            torch.optim.lr_scheduler._LRScheduler or None: Instance of the specified scheduler or
                                                            None if no scheduler is configured.
        
        Raises:
            ValueError: If the scheduler type is unsupported or required parameters are missing.
        """
        scheduler_config = self.config["lr_scheduler"]
        
        if scheduler_config["type"] == "":
            warnings.warn("No learning rate scheduler specified in the configuration, none will be used.")
            return None
    
        scheduler_type = scheduler_config["type"]
        scheduler_params = scheduler_config["params"]
        # Mapping of scheduler names to their corresponding PyTorch classes
        scheduler_mapping = {
            'StepLR': lr_scheduler.StepLR,
            'MultiStepLR': lr_scheduler.MultiStepLR,
            'ExponentialLR': lr_scheduler.ExponentialLR,
            'CosineAnnealingLR': lr_scheduler.CosineAnnealingLR,
            'CosineAnnealingWarmRestarts': lr_scheduler.CosineAnnealingWarmRestarts,
            'ReduceLROnPlateau': lr_scheduler.ReduceLROnPlateau,
            'LambdaLR': lr_scheduler.LambdaLR,
            'OneCycleLR': lr_scheduler.OneCycleLR,
            'CyclicLR': lr_scheduler.CyclicLR,
        }

        # Verify that the scheduler type is supported
        if scheduler_type not in scheduler_mapping:
            raise ValueError(f"Scheduler type '{scheduler_type}' is not supported. Please check the configuration.")
        scheduler_class = scheduler_mapping[scheduler_type]
        
        # Initialize the scheduler based on its type
        try:
                scheduler = scheduler_class(self.optimizer, **scheduler_params)
        except TypeError as e:
            raise ValueError(f"Error initializing '{scheduler_type}': {e}")

        
        logging.info(f"Scheduler '{scheduler_type}' initialized with parameters: {scheduler_params}")
        return scheduler

    def train(self, checkpoint_file=None):
        """Launch the training procedure of the Architect.
        
        Arguments:
            checkpoint_file: The path to the checkpoint file to load and resume a previous training. If None, the training will start from scratch."""
        use_cuda = "all" if self.device.type == "cuda" else None

        training_config = self.config["training"]
        self.max_epochs = training_config["max_epochs"]
        self.train_batch_size = training_config["train_batch_size"]
        self.patience = training_config["patience"]
        self.eval_interval = training_config["eval_interval"]
        self.eval_batch_size = training_config["eval_batch_size"]
        self.kg2nodetype = None
        # If we use a deep learning encoder, then we need to have HeteroData
        if self.config["model"]["encoder"]["name"] != "Default":
            logging.info("Creating Hetero Data from KG...")
            self.hetero_data, self.kg2het, self.het2kg, _, self.kg2nodetype = create_hetero_data(self.kg_train, self.config["metadata_csv"])
            self.hetero_data = self.hetero_data.to(self.device)


            self.node_embeddings = nn.ModuleDict()
            for node_type in self.hetero_data.node_types:
                num_nodes = self.hetero_data[node_type].num_nodes
                self.node_embeddings[node_type] = init_embedding(num_nodes, self.emb_dim, self.device)
        else:
            self.node_embeddings = init_embedding(self.kg_train.n_ent, self.emb_dim, self.device)
        self.rel_emb = init_embedding(self.kg_train.n_rel, self.emb_dim, self.device)

        logging.info("Initializing encoder...")
        self.encoder = self.initialize_encoder()

        logging.info("Initializing decoder...")
        self.decoder, self.criterion = self.initialize_decoder()
        self.decoder.to(self.device)

        logging.info("Initializing optimizer...")
        self.optimizer = self.initialize_optimizer()

        logging.info("Initializing sampler...")
        self.sampler = self.initialize_sampler()

        logging.info("Initializing lr scheduler...")
        self.scheduler = self.initialize_scheduler()

        self.training_metrics_file = Path(self.config["output_directory"], "training_metrics.csv")

        if checkpoint_file is None:
            with open(self.training_metrics_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Epoch", "Training Loss", "Validation MRR", "Learning Rate"])
        
        self.train_losses = []
        self.val_mrrs = []
        self.learning_rates = []

        train_iterator = DataLoader(self.kg_train, self.train_batch_size, use_cuda=use_cuda)
        logging.info(f"Number of training batches: {len(train_iterator)}")

        trainer = Engine(self.process_batch)
        RunningAverage(output_transform=lambda x: x).attach(trainer, "loss_ra")

        early_stopping = EarlyStopping(
            patience = self.patience,
            score_function = self.score_function,
            trainer = trainer
        )

        trainer.add_event_handler(Events.EPOCH_COMPLETED, self.log_metrics_to_csv)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, self.clean_memory)
        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=self.eval_interval), self.evaluate)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, self.update_scheduler)
        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=self.eval_interval), early_stopping)

        trainer.add_event_handler(Events.COMPLETED, self.on_training_completed)

        to_save = {
            "relations": self.rel_emb,
            "entities": self.node_embeddings,
            "decoder": self.decoder,
            "optimizer": self.optimizer,
            "trainer": trainer
        }
        if self.encoder.deep:
            to_save.update({"encoder":self.encoder})
        if self.scheduler is not None:
            to_save.update({"scheduler": self.scheduler})
        
        checkpoint_handler = Checkpoint(
            to_save,    # Dict of objects to save
            DiskSaver(dirname=self.checkpoints_dir, require_empty=False, create_dir=True), # Save manager
            n_saved=2,      # Only keep last 2 checkpoints
            global_step_transform=lambda *_: trainer.state.epoch     # Include epoch number
        )

        # Custom save function to move the model to CPU before saving and back to GPU after
        def save_checkpoint_to_cpu(engine):
            # Move models to CPU before saving
            if self.encoder.deep:
                self.encoder.to("cpu")
            self.decoder.to("cpu")
            self.rel_emb.to("cpu")
            self.node_embeddings.to("cpu")

            # Save the checkpoint
            checkpoint_handler(engine)

            # Move models back to GPU
            if self.encoder.deep:
               self.encoder.to(self.device)
            self.decoder.to(self.device)
            self.rel_emb.to(self.device)
            self.node_embeddings.to(self.device)

        # Attach checkpoint handler to trainer and call save_checkpoint_to_cpu
        trainer.add_event_handler(Events.EPOCH_COMPLETED, save_checkpoint_to_cpu)
    
        checkpoint_best_handler = ModelCheckpoint(
            dirname=self.checkpoints_dir,
            filename_prefix="best_model",
            n_saved=1,
            score_function=self.get_val_mrr,
            score_name="val_mrr",
            require_empty=False,
            create_dir=True,
            atomic=True
        )

        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=self.eval_interval),
            checkpoint_best_handler,
            to_save
        )


        if checkpoint_file is not None:
            if Path(checkpoint_file).is_file():
                logging.info(f"Resuming training from checkpoint: {checkpoint_file}")
                checkpoint = torch.load(checkpoint_file)
                Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

                logging.info("Checkpoint loaded successfully.")
                with open(self.training_metrics_file, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(['CHECKPOINT RESTART', 'CHECKPOINT RESTART', 'CHECKPOINT RESTART', 'CHECKPOINT RESTART'])

                if trainer.state.epoch < self.max_epochs:
                    logging.info(f"Starting from epoch {trainer.state.epoch}")
                    trainer.run(train_iterator)
                else:
                    logging.info(f"Training already completed. Last epoch is {trainer.state.epoch} and max_epochs is set to {self.max_epochs}")
            else:
                logging.info(f"Checkpoint file {checkpoint_file} does not exist. Starting training from scratch.")
                trainer.run(train_iterator, max_epochs=self.max_epochs)
        else:
            trainer.run(train_iterator, max_epochs=self.max_epochs)
    
        #################
        # Report metrics
        #################
        self.plot_learning_curves(self.training_metrics_file)

    def test(self):
        torch.cuda.empty_cache()
        gc.collect()

        logging.info("Loading best model.")
        best_model = find_best_model(self.checkpoints_dir)

        if not best_model:
            logging.error(f"No best model was found in {self.checkpoints_dir}. Make sure to run the training first and not rename checkpoint files before running evaluation.")
            return
        
        logging.info(f"Best model is {self.checkpoints_dir.joinpath(best_model)}")
        checkpoint = torch.load(self.checkpoints_dir.joinpath(best_model), map_location=self.device)
        self.load_state_dict(checkpoint["architect"])
        self.decoder.load_state_dict(checkpoint["decoder"])
        logging.info("Best model successfully loaded. \nEvaluating on the test set with best model...")

        self.decoder.eval()

        list_rel_1 = self.config["evaluation"]["made_directed_relations"]
        list_rel_2 = self.config["evaluation"]["target_relations"]
        thresholds = self.config["evaluation"]["thresholds"]
        mrr_file = Path(self.config["output_directory"], "evaluation_metrics.yaml")

        all_relations = set(self.kg_test.rel2ix.keys())
        remaining_relations = all_relations - set(list_rel_1) - set(list_rel_2)
        remaining_relations = list(remaining_relations)

        total_mrr_sum_list_1, fact_count_list_1, individual_mrrs_list_1, group_mrr_list_1 = self.calculate_mrr_for_relations(
            self.kg_test, list_rel_1)
        total_mrr_sum_list_2, fact_count_list_2, individual_mrrs_list_2, group_mrr_list_2 = self.calculate_mrr_for_relations(
            self.kg_test, list_rel_2)
        total_mrr_sum_remaining, fact_count_remaining, individual_mrrs_remaining, group_mrr_remaining = self.calculate_mrr_for_relations(
            self.kg_test, remaining_relations)

        global_mrr = (total_mrr_sum_list_1 + total_mrr_sum_list_2 + total_mrr_sum_remaining) / (fact_count_list_1 + fact_count_list_2 + fact_count_remaining)

        logging.info(f"Final Test MRR with best model: {global_mrr}")

        results = {
            "Global_MRR": global_mrr,
            "made_directed_relations": {
                "Global_MRR": group_mrr_list_1,
                "Individual_MRRs": individual_mrrs_list_1
            },
            "target_relations": {
                "Global_MRR": group_mrr_list_2,
                "Individual_MRRs": individual_mrrs_list_2
            },
            "remaining_relations": {
                "Global_MRR": group_mrr_remaining,
                "Individual_MRRs": individual_mrrs_remaining
            },
            "target_relations_by_frequency": {}  
        }

        for i in range(len(list_rel_2)):
            relation = list_rel_2[i]
            threshold = thresholds[i]
            frequent_indices, infrequent_indices = self.categorize_test_nodes(relation, threshold)
            frequent_mrr, infrequent_mrr = self.calculate_mrr_for_categories(frequent_indices, infrequent_indices)
            logging.info(f"MRR for frequent nodes (threshold={threshold}) in relation {relation}: {frequent_mrr}")
            logging.info(f"MRR for infrequent nodes (threshold={threshold}) in relation {relation}: {infrequent_mrr}")

            results["target_relations_by_frequency"][relation] = {
                "Frequent_MRR": frequent_mrr,
                "Infrequent_MRR": infrequent_mrr,
                "Threshold": threshold
            }
                
        
        with open(mrr_file, "w") as file:
            yaml.dump(results, file, default_flow_style=False, sort_keys=False)

        logging.info(f"Evaluation results stored in {mrr_file}")
        
    def infer(self, inference_kg_path):
        inference_mrr_file = Path(self.config["output_directory"], "inference_metrics.yaml")

        inference_df = pd.read_csv(inference_kg_path, sep="\t")
        inference_kg = KGATEGraph(df = inference_df, ent2ix=self.kg_train.ent2ix, rel2ix=self.kg_train.rel2ix) 
        
        # TODO : use node classification inference
        evaluator = KLinkPredictionEvaluator()
        evaluator.evaluate(self.eval_batch_size, 
                           self.decoder, inference_kg,
                           self.node_embeddings, self.rel_emb,
                           self.kg2nodetype, verbose=True)
            
        inference_mrr = evaluator.mrr()[1]
        inference_hit10 = evaluator.hit_at_k(10)[1]

        results = {"Inference MRR": inference_mrr, "Inference hit@10:": inference_hit10}

        logging.info(f"MRR on inference set: {inference_mrr}")

        with open(inference_mrr_file, "w") as file:
            yaml.dump(results, file, default_flow_style=False, sort_keys=False)


        logging.info(f"Evaluation results stored in {inference_mrr_file}")

    def process_batch(self, engine, batch):
        h, t, r = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
        n_h, n_t = self.sampler.corrupt_batch(h, t, r)
        n_h, n_t = n_h.to(self.device), n_t.to(self.device)

        self.optimizer.zero_grad()

        # Compute loss with positive and negative triples
        pos, neg = self.forward(h, t, r, n_h, n_t)
        loss = self.criterion(pos, neg)
        loss.backward()

        self.optimizer.step()

        self.normalize_parameters()

        return loss.item()

    def scoring_function(self, h_idx, t_idx, r_idx):
        encoder_output = None
        if self.encoder.deep:
            encoder_output = self.encoder.forward() #Check what the encoder needs
        
            h_node_types = [self.kg2nodetype[h.item()] for h in h_idx]
            t_node_types = [self.kg2nodetype[t.item()] for t in t_idx]
    
            try:
                h_het_idx = tensor([
                    self.kg2het[h_type][h.item()] for h, h_type in zip(h_idx, h_node_types)
                ], dtype=long, device=h_idx.device)
                t_het_idx = tensor([
                    self.kg2het[t_type][t.item()] for t, t_type in zip(t_idx, t_node_types)
                ], dtype=long, device=t_idx.device)
            except KeyError as e:
                logging.error(f"Mapping error on node ID: {e}")
                raise
        
            h_embeddings = stack([
                encoder_output[h_type][h_idx_item] for h_type, h_idx_item in zip(h_node_types, h_het_idx)
            ])
            t_embeddings = stack([
                encoder_output[t_type][t_idx_item] for t_type, t_idx_item in zip(t_node_types, t_het_idx)
            ])
        else:
            h_embeddings = self.node_embeddings(h_idx)
            t_embeddings = self.node_embeddings(t_idx)

        r_embeddings = self.rel_emb(r_idx)  # Relations are unchanged

        # Normalize entities (heads and tails)
        h_normalized = normalize(h_embeddings, p=2, dim=1)
        t_normalized = normalize(t_embeddings, p=2, dim=1)

        return self.decoder.score(h_normalized, r_embeddings, t_normalized)

    def normalize_parameters(self):
        # Some decoders should not normalize parameters or do so in a different way.
        # In this case, they should implement the function themselves and we return it.
        normalize_func = getattr(self.decoder, "normalize_parameters", None)
        # If the function only accept one parameter, it is the base torchKGE one,
        # we don't want that.
        if callable(normalize_func) and len(signature(normalize_func).parameters) > 1:
            stop_norm = normalize_func(rel_emb = self.rel_emb, ent_emb = self.node_embeddings)
            if stop_norm: return
        
        if self.encoder.deep:
            for node_type, embedding in self.node_embeddings.items():
                normalized_embedding = normalize(embedding.weight.data, p=2, dim=1)
                embedding.weight.data = normalized_embedding
                logging.debug(f"Normalized embeddings for node type '{node_type}'")
        else:
            self.node_embeddings.weight.data = normalize(self.node_embeddings.weight.data, p=2, dim=1)
        logging.debug(f"Normalized all embeddings")

        # Normaliser les embeddings des relations
        # self.rel_emb.weight.data = normalize(self.rel_emb.weight.data, p=2, dim=1)
        # logging.debug("Normalized relation embeddings")

    ##### Metrics recording in CSV file
    def log_metrics_to_csv(self, engine):
        epoch = engine.state.epoch
        train_loss = engine.state.metrics['loss_ra']
        val_mrr = engine.state.metrics.get('val_mrr', 0)
        lr = self.optimizer.param_groups[0]['lr']

        self.train_losses.append(train_loss)
        self.val_mrrs.append(val_mrr)
        self.learning_rates.append(lr)

        with open(self.training_metrics_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, train_loss, val_mrr, lr])

        logging.info(f"Epoch {epoch} - Train Loss: {train_loss}, Validation MRR: {val_mrr}, Learning Rate: {lr}")

    ##### Memory cleaning
    def clean_memory(self, engine):
        torch.cuda.empty_cache()
        gc.collect()
        logging.info("Memory cleaned.")

    ##### Evaluation on validation set
    def evaluate(self, engine):
        logging.info(f"Evaluating on validation set at epoch {engine.state.epoch}...")
        self.decoder.eval()  # Set the decoder to evaluation mode
        with torch.no_grad():
            # TODO : allow different task to be evaluated
            val_mrr = self.link_pred(self.kg_val) 
        engine.state.metrics['val_mrr'] = val_mrr 
        logging.info(f"Validation MRR: {val_mrr}")

        if self.scheduler and isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_mrr)
            logging.info('Stepping scheduler ReduceLROnPlateau.')

        self.decoder.train()  # Set the decoder back to training mode

    ##### Scheduler update
    def update_scheduler(self, engine):
        if self.scheduler is not None and not isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()

    ##### Early stopping score function
    def score_function(self, engine):
        val_mrr = engine.state.metrics.Get("val_mrr", 0)
        return val_mrr
    
    ##### Checkpoint best MRR
    def get_val_mrr(self, engine):
        return engine.state.metrics.get("val_mrr", 0)
    
    ##### Late stopping
    def on_training_completed(self, engine):
        logging.info(f"Training completed after {engine.state.epoch} epochs.")

    # TODO : create a script to isolate prediction functions. Maybe a Predictor class?
    def categorize_test_nodes(self, relation_name, threshold):
        """
        Categorizes test triples with the specified relation in the test set 
        based on whether their entities have been seen with that relation in the training set,
        and separates them into two groups based on a threshold for occurrences.

        Parameters
        ----------
        relation_name : str
            The name of the relation to check (e.g., 'indication').
        threshold : int
            The minimum number of occurrences of the relation for a node to be considered as "frequent".

        Returns
        -------
        frequent_indices : list
            Indices of triples in the test set with the specified relation where entities have been seen more than `threshold` times with that relation in the training set.
        infrequent_indices : list
            Indices of triples in the test set with the specified relation where entities have been seen fewer than or equal to `threshold` times with that relation in the training set.
        """
        # Get the index of the specified relation in the training graph
        if relation_name not in self.kg_train.rel2ix:
            raise ValueError(f"The relation '{relation_name}' does not exist in the training knowledge graph.")
        relation_idx = self.kg_train.rel2ix[relation_name]

        # Count occurrences of nodes with the specified relation in the training set
        train_node_counts = {}
        for i in range(self.kg_train.n_facts):
            if self.kg_train.relations[i].item() == relation_idx:
                head = self.kg_train.head_idx[i].item()
                tail = self.kg_train.tail_idx[i].item()
                train_node_counts[head] = train_node_counts.get(head, 0) + 1
                train_node_counts[tail] = train_node_counts.get(tail, 0) + 1

        # Separate test triples with the specified relation based on the threshold
        frequent_indices = []
        infrequent_indices = []
        for i in range(self.kg_test.n_facts):
            if self.kg_test.relations[i].item() == relation_idx:  # Only consider triples with the specified relation
                head = self.kg_test.head_idx[i].item()
                tail = self.kg_test.tail_idx[i].item()
                head_count = train_node_counts.get(head, 0)
                tail_count = train_node_counts.get(tail, 0)

                # Categorize based on threshold
                if head_count > threshold or tail_count > threshold:
                    frequent_indices.append(i)
                else:
                    infrequent_indices.append(i)

        return frequent_indices, infrequent_indices
    
    def calculate_mrr_for_relations(self, kg, relations):
        # MRR computed by ponderating for each relation
        mrr_sum = 0.0
        fact_count = 0
        individual_mrrs = {} 

        for relation_name in relations:
            # Get triples associated with index
            relation_index = kg.rel2ix.get(relation_name)
            indices_to_keep = torch.nonzero(kg.relations == relation_index, as_tuple=False).squeeze()

            if indices_to_keep.numel() == 0:
                continue  # Skip to next relation if no triples found
                
            print(relation_name)
            
            new_kg = kg.keep_triples(indices_to_keep)
            new_kg.dict_of_rels = kg.dict_of_rels
            new_kg.dict_of_heads = kg.dict_of_heads
            new_kg.dict_of_tails = kg.dict_of_tails
            test_mrr = self.link_pred(new_kg)
            
            # Save each relation's MRR
            individual_mrrs[relation_name] = test_mrr
            
            mrr_sum += test_mrr * indices_to_keep.numel()
            fact_count += indices_to_keep.numel()
        
        # Compute global MRR for the relation group
        group_mrr = mrr_sum / fact_count if fact_count > 0 else 0
        
        return mrr_sum, fact_count, individual_mrrs, group_mrr

    def calculate_mrr_for_categories(self, frequent_indices, infrequent_indices):
        """
        Calculate the MRR for frequent and infrequent categories based on given indices.
        
        Parameters
        ----------
        frequent_indices : list
            Indices of test triples considered as frequent.
        infrequent_indices : list
            Indices of test triples considered as infrequent.

        Returns
        -------
        frequent_mrr : float
            MRR for the frequent category.
        infrequent_mrr : float
            MRR for the infrequent category.
        """

        # Create subgraph for frequent and infrequent categories
        kg_frequent = self.kg_test.keep_triples(frequent_indices)
        kg_frequent.dict_of_rels = self.kg_test.dict_of_rels
        kg_frequent.dict_of_heads = self.kg_test.dict_of_heads
        kg_frequent.dict_of_tails = self.kg_test.dict_of_tails
        kg_infrequent = self.kg_test.keep_triples(infrequent_indices)
        kg_infrequent.dict_of_rels = self.kg_test.dict_of_rels
        kg_infrequent.dict_of_heads = self.kg_test.dict_of_heads
        kg_infrequent.dict_of_tails = self.kg_test.dict_of_tails
        
        # Compute each category's MRR
        frequent_mrr = self.link_pred(kg_frequent) if frequent_indices else 0
        infrequent_mrr = self.link_pred(kg_infrequent) if infrequent_indices else 0

        return frequent_mrr, infrequent_mrr

    def link_pred(self, kg):
        """Link prediction evaluation on test set."""
        # Test MRR measure
        evaluator = KLinkPredictionEvaluator()
        evaluator.evaluate(self.eval_batch_size,
                           self.decoder, kg,
                           self.node_embeddings, self.rel_emb,
                           self.kg2nodetype, verbose=True)
        
        test_mrr = evaluator.mrr()[1]
        return test_mrr