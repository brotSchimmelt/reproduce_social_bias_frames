import logging
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from icecream import ic
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup

import config
from utils.helper import get_device, get_run_time
from utils.SBICDataset import SBICDataset

#### LOGGING ####
if not config.IC_ENABLE:
    ic.disable()

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

    # add new and special tokens
    tokenizer.add_special_tokens(
        {
            "pad_token": "[PAD]",
            "bos_token": config.START_TOKEN,
            "eos_token": config.END_TOKEN,
            "additional_special_tokens": [config.SEP_TOKEN],
        }
    )
    tokenizer.add_tokens(config.OTHER_TOKENS)

    return tokenizer


def validate(dev_data: DataLoader, model: GPT2LMHeadModel, device: str) -> None:
    """Validate the performance on the dev dataset."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, a in tqdm(dev_data):
            X = X.to(device)
            a = a.to(device)
            outputs = model(X, attention_mask=a, labels=X)
            loss = outputs.loss
            total_loss += loss.item()

    avg_val_loss = total_loss / len(dev_data)
    logging.info(f"Validation Loss: {avg_val_loss}")
    print(f"Validation Loss: {avg_val_loss}")
    model.train()


def train(
    train_data: DataLoader,
    dev_data: DataLoader,
    model: GPT2LMHeadModel,
    epochs: int = config.EPOCHS,
) -> None:
    """Train the model and validate on dev set."""

    # move model to GPU
    device = get_device()
    model.to(device)
    logging.info(f"Device: {device}")

    model.train()

    # the paper mentions linear warmup
    optim = Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optim, config.WARMUP_STEPS, len(train_data) * epochs
    )

    # train loop
    for idx in range(epochs):
        print(f"Epoch {idx+1}")
        total_loss = 0
        for X, a in tqdm(train_data):
            X = X.to(device)
            a = a.to(device)
            optim.zero_grad()
            loss = model(X, attention_mask=a, labels=X).loss
            loss.backward()
            optim.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_data)
        logging.info(f"Epoch {idx+1} - Training Loss: {avg_train_loss}")
        print(f"Epoch {idx+1} - Training Loss: {avg_train_loss}")

    validate(dev_data, model, device)


def main() -> None:

    # load model and tokenizer
    print("Loading model and tokenizer...")
    model_path = config.GPT2_SMALL
    tokenizer = load_tokenizer(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))  # since we added new tokens

    # load datasets
    print("Loading datasets...")
    train_data = SBICDataset(config.SBIC_TRAIN_PATH, tokenizer)
    dev_data = SBICDataset(config.SBIC_DEV_PATH, tokenizer)
    train_dataset = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    dev_dataset = DataLoader(dev_data, batch_size=config.BATCH_SIZE, shuffle=False)

    print("Training model...")
    train(train_dataset, dev_dataset, model)


if __name__ == "__main__":
    main()
    print(f"Elapsed time: {get_run_time(start)} min.")
