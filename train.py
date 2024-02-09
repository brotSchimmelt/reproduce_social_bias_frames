import logging
import os
import shutil
from datetime import datetime

import torch
import torch.nn.functional as F
from icecream import ic
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup

import config
from utils.helper import get_device, get_run_time, set_seed
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
            # "additional_special_tokens": [config.SEP_TOKEN],
            "additional_special_tokens": config.OTHER_TOKENS + [config.SEP_TOKEN],
        }
    )
    # tokenizer.add_tokens(config.OTHER_TOKENS)

    return tokenizer


def validate(
    dev_data: DataLoader, model: GPT2LMHeadModel, device: str, is_test_set: bool = False
) -> float:
    """Validate the performance on the dev dataset."""

    model.eval()

    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dev_data):

            # unpack batch
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["input_ids"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

    # compute average validation loss
    avg_val_loss = total_loss / len(dev_data)
    if is_test_set:
        logging.info(f"Test Loss: {avg_val_loss}")
        print(f"Test Loss: {avg_val_loss}")
    else:
        logging.info(f"Validation Loss: {avg_val_loss}")
        print(f"Validation Loss: {avg_val_loss}")

    model.train()

    return avg_val_loss


def train(
    train_data: DataLoader,
    dev_data: DataLoader,
    model: GPT2LMHeadModel,
    epochs: int = config.EPOCHS,
) -> GPT2LMHeadModel:
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
    val_loss_history, model_checkpoint_paths = [], []

    # train loop
    for idx in range(epochs):
        print(f"Start Epoch {idx+1}:")
        logging.info(f"Start Epoch {idx+1}:")
        total_loss = 0
        for batch in tqdm(train_data):

            # unpack batch
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["input_ids"].to(
                device
            )  # labels are same as input_ids for language modeling

            optim.zero_grad()
            output = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss
            loss.backward()

            optim.step()
            scheduler.step()

            total_loss += loss.item()

        # compute average training loss for one epoch
        avg_train_loss = total_loss / len(train_data)
        logging.info(f"Epoch {idx+1} - Training Loss: {avg_train_loss}")
        print(f"Epoch {idx+1} - Training Loss: {avg_train_loss}")

        val_loss = validate(dev_data, model, device)

        # save current model version and val loss
        val_loss_history.append(val_loss)
        epoch_model_path = f"tmp/early_stopping/intermediate_model_epoch_{idx+1}.pt"
        torch.save(model.state_dict(), epoch_model_path)
        model_checkpoint_paths.append(epoch_model_path)

    # save only the best model
    best_epoch = val_loss_history.index(min(val_loss_history)) + 1
    best_model_path = model_checkpoint_paths[best_epoch - 1]
    best_model_final_path = f"tmp/models/best_model_{epochs}_{timestamp}.pt"
    shutil.copy(best_model_path, best_model_final_path)
    logging.info(f"Best model saved at: {best_model_final_path}")
    logging.info(f"Best model epoch: {best_epoch}")
    logging.info(f"Best model val loss: {val_loss_history[best_epoch-1]}")

    # cleanup
    for path in model_checkpoint_paths:
        os.remove(path)

    model.load_state_dict(torch.load(best_model_final_path))
    return model


def main() -> None:

    # load model and tokenizer
    print("Loading model and tokenizer...")
    model_path = config.MODEL_TYPE
    tokenizer = load_tokenizer(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))  # since we added new tokens

    # load datasets
    print("Loading datasets...")
    train_data = SBICDataset(config.SBIC_TRAIN_PATH, tokenizer)
    dev_data = SBICDataset(config.SBIC_DEV_PATH, tokenizer)
    test_data = SBICDataset(config.SBIC_TEST_PATH, tokenizer)
    train_dataset = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    dev_dataset = DataLoader(dev_data, batch_size=config.BATCH_SIZE, shuffle=False)
    test_dataset = DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=False)

    print("Training model...")
    trained_model = train(train_dataset, dev_dataset, model)

    # check performance on the test set
    _ = validate(test_dataset, trained_model, get_device(), is_test_set=True)
    logging.info("Done.")


if __name__ == "__main__":
    set_seed(config.SEED)
    main()
    logging.info(f"Training took {get_run_time(start)} minutes")
    print(f"Elapsed time: {get_run_time(start)} min.")
