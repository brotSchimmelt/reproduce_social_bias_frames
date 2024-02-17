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
from utils.helper import get_run_time
from utils.SBICDataset import SBICDataset

#### LOGGING ####
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


def load_data(tokenizer: GPT2Tokenizer) -> Tuple[SBICDataset, SBICDataset, SBICDataset]:
    """Load the SBIC dataset."""
    train = SBICDataset(config.SBIC_TRAIN_PATH, tokenizer)
    dev = SBICDataset(config.SBIC_DEV_PATH, tokenizer)
    test = SBICDataset(config.SBIC_TEST_PATH, tokenizer)
    logging.info(f"Split: train={len(train)}, dev={len(dev)}, test={len(test)}")
    return train, dev, test


def train(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer) -> GPT2LMHeadModel:
    print("Loading datasets...")
    train_data, dev_data, test_data = load_data(tokenizer)

    training_args = TrainingArguments(
        output_dir="./tmp/results",
        num_train_epochs=config.DEFAULT_NUM_EPOCHS,  # 1,2,5
        per_device_train_batch_size=config.DEFAULT_BATCH_SIZE,  # 4
        per_device_eval_batch_size=config.DEFAULT_BATCH_SIZE,  # 4
        warmup_steps=config.DEFAULT_WARMUP_STEPS,  # number of warmup steps is not mentioned in the paper
        learning_rate=config.DEFAULT_LEARNING_RATE,  # 1e-5
        logging_dir="./logs",
        logging_steps=config.LOGGING_STEPS,  # 5_000
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
    )

    print("Training model...")
    trainer.train()

    # final evaluation
    test_results = trainer.evaluate(test_data)
    print(f"Test Loss: {test_results['eval_loss']}")
    logging.info(f"Test Loss: {test_results['eval_loss']}")

    return model


def main(model_path: str) -> None:
    print("Loading model and tokenizer...")
    tokenizer = load_tokenizer(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))  # since we added new tokens
    logging.info(f"Model and tokenizer loaded from {model_path}")

    # train the model
    trained_model = train(model, tokenizer)

    # save the model and the tokenizer
    now = datetime.now().strftime(config.LOGGING_FILE_DATEFMT)
    model_save_path = f"./tmp/models/best_model_{now}"
    trained_model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    logging.info(f"Model and tokenizer saved to {model_save_path}")


if __name__ == "__main__":
    set_seed(config.DEFAULT_SEED)
    main(config.GPT2_XL)
    print(f"Elapsed time: {get_run_time(start)} min.")
