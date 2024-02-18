import argparse
import logging
import os
from datetime import datetime
from typing import Tuple

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

import config
from utils.helper import EvalLoggingCallback, get_run_time
from utils.SBICDataset import SBICDataset

#### LOGGING ####
if not os.path.exists(config.LOG_DIR):
    os.makedirs(config.LOG_DIR)
app_name = os.path.splitext(os.path.basename(__file__))[0]
start = datetime.now()
timestamp = start.strftime(config.LOGGING_FILE_DATEFMT)
logging.basicConfig(
    filename=f"{config.LOG_DIR}{app_name}_{timestamp}.log",
    level=config.LOGGING_LEVEL,
    format=config.LOGGING_FORMAT,
    datefmt=config.LOGGING_DATEFMT,
)
#### LOGGING ####


def parse_arguments() -> argparse.Namespace:
    """Handle the command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a GPT2 model on the SBIC dataset."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=config.DEFAULT_MODEL,
        help="Path to the pre-trained model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.DEFAULT_NUM_EPOCHS,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config.DEFAULT_LEARNING_RATE,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=config.DEFAULT_WARMUP_STEPS,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config.DEFAULT_BATCH_SIZE,
        help="Training and evaluation batch size",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=config.DEFAULT_SEED,
        help="Random seed for initialization",
    )
    parser.add_argument(
        "--use_fill_token",
        action="store_true",  # default=False
        help="Whether to use a fill token or not",
    )
    return parser.parse_args()


def load_tokenizer(model_path: str) -> GPT2Tokenizer:
    """Load tokenizer and add special tokens."""
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    # add new special tokens
    tokenizer.add_special_tokens(
        {
            "pad_token": config.PAD_TOKEN,
            "bos_token": config.START_TOKEN,
            "eos_token": config.END_TOKEN,
            "additional_special_tokens": config.OTHER_TOKENS + [config.SEP_TOKEN],
        }
    )
    return tokenizer


def load_data(
    tokenizer: GPT2Tokenizer, fill_token_flag: bool
) -> Tuple[SBICDataset, SBICDataset, SBICDataset]:
    """Load the SBIC dataset."""
    train = SBICDataset(config.SBIC_TRAIN_PATH, tokenizer, fill_token_flag)
    dev = SBICDataset(config.SBIC_DEV_PATH, tokenizer, fill_token_flag)
    test = SBICDataset(config.SBIC_TEST_PATH, tokenizer, fill_token_flag)
    logging.info(f"Split: train={len(train)}, dev={len(dev)}, test={len(test)}")
    return train, dev, test


def train(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    model_name: str,
    args: argparse.Namespace,
) -> GPT2LMHeadModel:
    print("Loading datasets...")
    train_data, dev_data, test_data = load_data(tokenizer, args.use_fill_token)

    output_dir = (
        config.CHECKPOINT_DIR
        + f"{model_name}_{'fillY' if args.use_fill_token else 'fillN'}_{args.random_seed}/"
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,  # 1,2,5
        per_device_train_batch_size=args.batch_size,  # 4
        per_device_eval_batch_size=args.batch_size,  # 4
        warmup_steps=args.warmup_steps,  # number of warmup steps is not mentioned in the paper
        learning_rate=args.lr,  # 1e-5
        logging_dir=config.LOG_DIR,  # logs/
        logging_steps=config.LOGGING_STEPS,  # 500
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
        callbacks=[EvalLoggingCallback(output_dir=output_dir + "validation.log")],
    )

    print("Training model...")
    trainer.train()

    # final evaluation
    test_results = trainer.evaluate(test_data)
    print(f"Test Loss: {test_results['eval_loss']}")
    logging.info(f"Test Loss: {test_results['eval_loss']}")

    return model


def main() -> None:
    # get the command line arguments and set the random seed
    args = parse_arguments()
    set_seed(args.random_seed)

    print("Loading model and tokenizer...")
    model_name = args.model_path.split("/")[-1]
    tokenizer = load_tokenizer(args.model_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.resize_token_embeddings(len(tokenizer))  # since we added new tokens
    logging.info(f"Model and tokenizer loaded from {args.model_path}")

    # train the model
    trained_model = train(model, tokenizer, model_name=model_name, args=args)

    # save the model and the tokenizer
    now = datetime.now().strftime(config.LOGGING_FILE_DATEFMT)
    model_save_path = f"./tmp/models/best_model_fillY_{args.random_seed}_{now}"
    trained_model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    logging.info(f"Model and tokenizer saved to {model_save_path}")


if __name__ == "__main__":
    main()
    print(f"Elapsed time: {get_run_time(start)} min.")
    logging.info(f"Elapsed time: {get_run_time(start)} min.")
