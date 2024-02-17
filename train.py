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
        num_train_epochs=config.EPOCHS,  # 1,2,5
        per_device_train_batch_size=config.BATCH_SIZE,  # 4
        per_device_eval_batch_size=config.BATCH_SIZE,  # 4
        warmup_steps=config.WARMUP_STEPS,  # number of warmup steps is not mentioned in the paper
        learning_rate=config.LEARNING_RATE,  # 1e-5
        logging_dir="./logs",
        logging_steps=config.LOGGING_STEPS,
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


def main() -> None:
    print("Loading model and tokenizer...")
    model_path = config.MODEL_TYPE
    tokenizer = load_tokenizer(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))  # since we added new tokens

    # train the model
    trained_model = train(model, tokenizer)

    # save the model and the tokenizer
    model_path = f"./tmp/models/best_model_{timestamp}"
    trained_model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    logging.info(f"Model and tokenizer saved to {model_path}")


if __name__ == "__main__":
    set_seed(config.SEED)
    main()
    print(f"Elapsed time: {get_run_time(start)} min.")
