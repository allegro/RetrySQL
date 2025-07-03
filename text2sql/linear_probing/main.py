import argparse
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from text2sql.linear_probing.data import LinearProbingDataModule
from text2sql.linear_probing.model import LinearProbingModel
from text2sql.llm_training.domain.model import LinearProbingConfiguration
from text2sql.commons.logging_utils import get_logger

logger = get_logger(__name__)

def parse_args() -> LinearProbingConfiguration:
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument("--backbone_model_name_or_path", type=str, default="infly/OpenCoder-1.5B-Base", help="Path to the backbone model")
    parser.add_argument("--tokenizer_name_or_path", type=str, default="infly/OpenCoder-1.5B-Base", help="Path to the tokenizer")
    parser.add_argument("--path_to_dataset", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--num_sanity_val_steps", type=int, default=0, help="Number of sanity validation steps")
    parser.add_argument("--log_every_n_steps", type=int, default=1, help="Logging frequency in steps")
    parser.add_argument("--val_check_interval", type=float, default=0.25, help="Validation check interval")
    parser.add_argument("--train_val_split", type=float, default=0.8, help="Define the ratio of train/val data")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for data loading")

    args = parser.parse_args()
    seed_everything(args.seed)

    return LinearProbingConfiguration(
        backbone_model_name_or_path=args.backbone_model_name_or_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        path_to_dataset=args.path_to_dataset,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        num_sanity_val_steps=args.num_sanity_val_steps,
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=args.val_check_interval,
        train_val_split=args.train_val_split,
        num_workers=args.num_workers
    )

if __name__ == '__main__':
    config = parse_args()
    data_module = LinearProbingDataModule(config)
    model = LinearProbingModel(config)

    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        num_sanity_val_steps=config.num_sanity_val_steps,
        log_every_n_steps=config.log_every_n_steps,
        val_check_interval=config.val_check_interval
    )
    trainer.fit(model, data_module)
