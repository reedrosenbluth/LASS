import argparse
import logging
import os
import pathlib
from typing import List, NoReturn
import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import WandbLogger
from torch.utils.tensorboard import SummaryWriter
from data.datamodules import *
from data.precomputed_stft_dataset import PrecomputedSTFTDataset
from utils import create_logging, parse_yaml
from models.resunet import *
from losses import get_loss_function
from models.audiosep import AudioSep, get_model_class
from models.clap_encoder import CLAP_Encoder
from callbacks.base import CheckpointEveryNSteps
from optimizers.lr_schedulers import get_lr_lambda
import wandb


def get_dirs(
    workspace: str, 
    filename: str, 
    config_yaml: str, 
    devices_num: int
) -> List[str]:
    r"""Get directories and paths.

    Args:
        workspace (str): directory of workspace
        filename (str): filename of current .py file.
        config_yaml (str): config yaml path
        devices_num (int): 0 for cpu and 8 for training with 8 GPUs

    Returns:
        checkpoints_dir (str): directory to save checkpoints
        logs_dir (str), directory to save logs
        tf_logs_dir (str), directory to save TensorBoard logs
        statistics_path (str), directory to save statistics
    """
    
    os.makedirs(workspace, exist_ok=True)

    yaml_name = pathlib.Path(config_yaml).stem

    # Directory to save checkpoints
    checkpoints_dir = os.path.join(
        workspace,
        "checkpoints",
        filename,
        "{},devices={}".format(yaml_name, devices_num),
    )
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Directory to save logs
    logs_dir = os.path.join(
        workspace,
        "logs",
        filename,
        "{},devices={}".format(yaml_name, devices_num),
    )
    os.makedirs(logs_dir, exist_ok=True)

    # Directory to save TensorBoard logs
    create_logging(logs_dir, filemode="w")
    logging.info(args)

    tf_logs_dir = os.path.join(
        workspace,
        "tf_logs",
        filename,
        "{},devices={}".format(yaml_name, devices_num),
    )

    # Directory to save statistics
    statistics_path = os.path.join(
        workspace,
        "statistics",
        filename,
        "{},devices={}".format(yaml_name, devices_num),
        "statistics.pkl",
    )
    os.makedirs(os.path.dirname(statistics_path), exist_ok=True)

    return checkpoints_dir, logs_dir, tf_logs_dir, statistics_path

 
def get_data_module(
    config_yaml: str,
    num_workers: int,
    batch_size: int,
) -> DataModule:
    r"""Create data_module using PrecomputedSTFTDataset with LMDB.

    Args:
        config_yaml: str
        num_workers: int
        batch_size: int

    Returns:
        data_module: DataModule
    """

    # read configurations
    configs = parse_yaml(config_yaml)
    precomputed_stft_dir = configs['data'].get('precomputed_stft_dir', None)

    logging.info("Using precomputed STFT dataset from LMDB.")
    if not precomputed_stft_dir:
        raise ValueError("Config key 'data.precomputed_stft_dir' must be set (base directory containing train.lmdb and val.lmdb).")

    base_lmdb_path = pathlib.Path(precomputed_stft_dir)
    precomputed_train_path = base_lmdb_path / "train.lmdb"
    precomputed_val_path = base_lmdb_path / "val.lmdb"

    logging.info(f"Loading precomputed training data from LMDB: {precomputed_train_path}")
    logging.info(f"Loading precomputed validation data from LMDB: {precomputed_val_path}")

    # Instantiate the precomputed datasets for training and validation
    # lock=False is recommended for multi-worker DataLoaders
    train_dataset = PrecomputedSTFTDataset(
        data_path=str(precomputed_train_path),
        lock=False
    )

    # val_dataset = PrecomputedSTFTDataset(
    #     data_path=str(precomputed_val_path),
    #     lock=False
    # )

    # data module
    # IMPORTANT: Assumes DataModule accepts train_dataset and val_dataset arguments.
    # If your DataModule definition in data/datamodules.py is different, it needs modification.
    data_module = DataModule(
        train_dataset=train_dataset,
        # val_dataset=val_dataset, # Added validation dataset
        num_workers=num_workers,
        batch_size=batch_size
    )

    return data_module


def train(args) -> NoReturn:
    r"""Train, evaluate, and save checkpoints.

    Args:
        workspace: str, directory of workspace
        gpus: int, number of GPUs to train
        config_yaml: str
    """

    # arguments & parameters
    workspace = args.workspace
    config_yaml = args.config_yaml
    filename = args.filename

    devices_num = torch.cuda.device_count()
    # Read config file.
    configs = parse_yaml(config_yaml)

    # Configuration of the separation model
    query_net = configs['model']['query_net']
    model_type = configs['model']['model_type']
    input_channels = configs['model']['input_channels']
    output_channels = configs['model']['output_channels']
    condition_size = configs['model']['condition_size']
    use_text_ratio = configs['model']['use_text_ratio']
    
    # Configuration of the trainer
    num_nodes = configs['train']['num_nodes']
    batch_size = configs['train']['batch_size_per_device'] 
    sync_batchnorm = configs['train']['sync_batchnorm'] 
    num_workers = configs['train']['num_workers']
    loss_type = configs['train']['loss_type']
    optimizer_type = configs["train"]["optimizer"]["optimizer_type"]
    learning_rate = float(configs['train']["optimizer"]['learning_rate'])
    lr_lambda_type = configs['train']["optimizer"]['lr_lambda_type']
    warm_up_steps = configs['train']["optimizer"]['warm_up_steps']
    reduce_lr_steps = configs['train']["optimizer"]['reduce_lr_steps']
    save_step_frequency = configs['train']['save_step_frequency']
    resume_checkpoint_path = args.resume_checkpoint_path
    if resume_checkpoint_path == "":
        resume_checkpoint_path = None
    else:
        logging.info(f'Finetuning AudioSep with checkpoint [{resume_checkpoint_path}]')

    # Get directories and paths
    checkpoints_dir, logs_dir, tf_logs_dir, statistics_path = get_dirs(
        workspace, filename, config_yaml, devices_num,
    )

    logging.info(configs)

    # data module
    data_module = None # Initialize
    try:
        data_module = get_data_module(
            config_yaml=config_yaml,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        
        # model
        Model = get_model_class(model_type=model_type)

        ss_model = Model(
            input_channels=input_channels,
            output_channels=output_channels,
            condition_size=condition_size,
        )

        # loss function
        loss_function = get_loss_function(loss_type)

        # SegmentMixer is no longer needed as mixing/STFT is precomputed
        segment_mixer = None 
        logging.info("SegmentMixer skipped as precomputed STFTs are used.")

        
        if query_net == 'CLAP':
            query_encoder = CLAP_Encoder()
        else:
            raise NotImplementedError

        lr_lambda_func = get_lr_lambda(
            lr_lambda_type=lr_lambda_type,
            warm_up_steps=warm_up_steps,
            reduce_lr_steps=reduce_lr_steps,
        )

        # pytorch-lightning model
        pl_model = AudioSep(
            ss_model=ss_model,
            query_encoder=query_encoder,
            loss_function=loss_function,
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            lr_lambda_func=lr_lambda_func,
            use_text_ratio=use_text_ratio,
        )

        checkpoint_every_n_steps = CheckpointEveryNSteps(
            checkpoints_dir=checkpoints_dir,
            save_step_frequency=save_step_frequency,
        )

        summary_writer = SummaryWriter(log_dir=tf_logs_dir)

        yaml_name = pathlib.Path(config_yaml).stem

        wandb_logger = WandbLogger(
            project="LASS",
            name=f"{yaml_name}_devices{devices_num}",
            save_dir=tf_logs_dir,
            log_model=True,
            config=configs,
        )

        callbacks = [checkpoint_every_n_steps]

        trainer = pl.Trainer(
            accelerator='auto',
            devices='auto',
            strategy='ddp_find_unused_parameters_true',
            num_nodes=num_nodes,
            precision="32-true",
            logger=wandb_logger,
            callbacks=callbacks,
            fast_dev_run=False,
            max_epochs=-1,
            log_every_n_steps=50,
            use_distributed_sampler=True,
            sync_batchnorm=sync_batchnorm,
            num_sanity_val_steps=0,
            enable_checkpointing=False,
            enable_progress_bar=True,
            enable_model_summary=True,
        )

        # Fit, evaluate, and save checkpoints.
        trainer.fit(
            model=pl_model, 
            train_dataloaders=None,
            val_dataloaders=None,
            datamodule=data_module,
            ckpt_path=resume_checkpoint_path,
        )

    finally:
        # Ensure datasets are closed properly after training/evaluation
        if data_module is not None:
            logging.info("Closing datasets...")
            # Check if datasets exist before attempting to close
            if hasattr(data_module, 'train_dataset') and data_module.train_dataset:
                try:
                    data_module.train_dataset.close()
                    logging.info("Closed training dataset.")
                except Exception as e:
                    logging.error(f"Error closing training dataset: {e}")
            if hasattr(data_module, 'val_dataset') and data_module.val_dataset:
                 try:
                    data_module.val_dataset.close()
                    logging.info("Closed validation dataset.")
                 except Exception as e:
                    logging.error(f"Error closing validation dataset: {e}")
        
        if wandb.run:
             wandb.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace", type=str, required=True, help="Directory of workspace."
    )
    parser.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="Path of config file for training.",
    )

    parser.add_argument(
        "--resume_checkpoint_path",
        type=str,
        required=True,
        default='',
        help="Path of pretrained checkpoint for finetuning.",
    )

    args = parser.parse_args()
    args.filename = pathlib.Path(__file__).stem

    train(args)
