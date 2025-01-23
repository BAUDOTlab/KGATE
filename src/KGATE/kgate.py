import os
from pathlib import Path
from torchkge import Model
import torchkge.sampling as sampling
import torch
from torch import tensor, long, stack
from .utils import parse_config, load_knowledge_graph, set_random_seeds
from .preprocessing import prepare_knowledge_graph
from .encoders import *
from .decoders import *
from torchkge.utils import MarginLoss, BinaryCrossEntropyLoss, DataLoader
import logging
import warnings
import torch.optim as optim
from torch.optim import lr_scheduler
import csv
from ignite.metrics import RunningAverage
from ignite.engine import Events, Engine
from ignite.handlers import EarlyStopping, ModelCheckpoint, Checkpoint, DiskSaver

# Configure logging
logging.captureWarnings(True)
log_level = logging.INFO# if config["common"]['verbose'] else logging.WARNING
logging.basicConfig(
    level=log_level,  
    format='%(asctime)s - %(levelname)s - %(message)s' 
)

TRANSLATIONAL_MODELS = ['TransE', 'TransH', 'TransR', 'TransD', 'TorusE']

class Architect(Model):
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

        # If we use a deep learning encoder, then we need to have HeteroData
        if self.config["model"]["encoder"]["name"] != "Default":
            self.hetero_data, self.kg2het, self.het2kg, _, self.kg2nodetype = create_hetero_data(kg, self.config["mapping_csv"])
            self.hetero_data = self.hetero_data.to(device)

            self.node_embeddings = nn.ModuleDict()
            for node_type in self.hetero_data.node_types:
                num_nodes = self.hetero_data[node_type].num_nodes
                self.node_embeddings[node_type] = my_init_embedding(num_nodes, self.emb_dim)
        else:
            self.node_embeddings = self.init_embedding(self.kg_train.n_ent, self.emb_dim)
        self.rel_emb = self.init_embedding(self.kg_train.n_rel, self.emb_dim)

        logging.info("Initializing encoder...")
        self.encoder = self.initialize_encoder()

        logging.info("Initializing decoder...")
        self.decoder, self.criterion = self.initialize_decoder()
        self.decoder.to(device)

        logging.info("Initializing optimizer...")
        self.optimizer = self.initialize_optimizer()

        logging.info("Initializing sampler...")
        self.sampler = self.initialize_sampler()

        logging.info("Initializing lr scheduler...")
        self.scheduler = self.initialize_scheduler()

        self.training_metrics_file = Path(config["output_directory"], "training_metrics.csv")
        training_config = self.config["training"]
        if not config["resume_checkpoint"] and run_training:
            with open(self.training_metrics_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Epoch", "Training Loss", "Validation MRR", "Learning Rate"])
        
        self.train_losses = []
        self.val_mrrs = []
        self.learning_rates = []

        self.max_epochs = training_config["max_epochs"]
        self.batch_size = training_config["batch_size"]
        self.patience = training_config["patience"]
        self.eval_interval = training_config["eval_interval"]
        self.eval_batch_size = training_config["eval_batch_size"]

    def initialize_encoder(self):
        mapping_csv = self.config["mapping_csv"]
        encoder_config = self.config["model"]["encoder"]
        encoder_name = encoder_config["name"]
        gnn_layers = encoder_config["gnn_layer_number"]

        match encoder_name:
            case "Default":
                encoder = DefaultEncoder(self.kg_train.n_ent, self.kg_train.n_rel, self.emb_dim)

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
                sampler = sampling.PositionalNegativeSampler(self.kg_train, self.kg_val, self.kg_test)
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

    def init_embedding(self, num_embeddings, emb_dim):
        embedding = nn.Embedding(num_embeddings, emb_dim)
        nn.init.xavier_uniform_(embedding.weight.data)
        return embedding

    def train(self):
        use_cuda = "all" if self.device.type == "cuda" else None
        train_iterator = DataLoader(self.kg_train, self.batch_size, use_cuda=use_cuda)
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

        if self.scheduler is not None:
            to_save = {
                "architect": self,
                "encoder": self.encoder,
                "decoder": self.decoder,
                "optimizer": self.optimizer,
                "scheduler": self.scheduler,
                "trainer": trainer
            }
        else:
            to_save = {
                "architect": self,
                "encoder": self.encoder,
                "decoder": self.decoder,
                "optimizer": self.optimizer,
                "trainer": trainer
            }
        
        checkpoints_dir = Path(self.config["output_directory"], "checkpoints")

        checkpoint_handler = Checkpoint(
            to_save,    # Dict of objects to save
            DiskSaver(dirname=checkpoints_dir, require_empty=False, create_dir=True), # Save manager
            n_saved=2,      # Only keep last 2 checkpoints
            global_step_transform=lambda *_: trainer.state.epoch     # Include epoch number
        )

        # Custom save function to move the model to CPU before saving and back to GPU after
        def save_checkpoint_to_cpu(engine):
            # Move model to CPU before saving
            self.to('cpu')
            self.encoder.to("cpu")
            self.decoder.to('cpu')

            # Save the checkpoint
            checkpoint_handler(engine)

            # Move model back to GPU
            self.to(self.device)
            self.encoder.to(self.device)
            self.decoder.to(self.device)

        # Attach checkpoint handler to trainer and call save_checkpoint_to_cpu
        trainer.add_event_handler(Events.EPOCH_COMPLETED, save_checkpoint_to_cpu)
    


    def process_batch(self, engine, batch):
        h, t, r = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
        n_h, n_t = self.sampler.corrupt_batch(h, t, r)
        n_h, n_t = n_h.to(self.device), n_t.to(self.device)

        self.optimizer.zero_gard()

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
            h_embeddings = self.ent_emb(h_idx)
            t_embeddings = self.ent_emb(t_idx)

        r_embeddings = self.rel_emb(r_idx)  # Relations are unchanged

        # Normalize entities (heads and tails)
        h_normalized = normalize(h_embeddings, p=2, dim=1)
        t_normalized = normalize(t_embeddings, p=2, dim=1)

        return self.decoder.score(h_normalized, r_embeddings, t_normalized)

    def normalize_parameters(self):
        # Some decoders should not normalize parameters or do so in a different way.
        # In this case, they should implement the function themselves and we return it.
        normalize_func = getattr(self.decoder, "normalize_parameters", None)
        if callable(normalize_func):
            stop_norm = normalize_func(self)
            if stop_norm: return
        
        if self.encoder.deep:
            for node_type, embedding in self.node_embeddings.items():
                normalized_embedding = normalize(embedding.weight.data, p=2, dim=1)
                embedding.weight.data = normalized_embedding
        else:
            self.node_embeddings.weight.data = normalize(self.node_embeddings.weight.data, p=2, dim=1)
        logging.debug(f"Normalized embeddings for node type '{node_type}'")

        # Normaliser les embeddings des relations
        self.rel_emb.weight.data = normalize(self.rel_emb.weight.data, p=2, dim=1)
        logging.debug("Normalized relation embeddings")

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
            val_mrr = link_pred(self.decoder, self.kg_val, self.eval_batch_size) 
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