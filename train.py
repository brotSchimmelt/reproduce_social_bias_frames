import logging
import os
import shutil
from datetime import datetime
from typing import Tuple

import torch
from icecream import ic
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup

import config
from utils.helper import get_device, get_run_time, set_seed, write_output_to_csv
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

    # add new special tokens
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


def load_data(tokenizer: GPT2Tokenizer) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load the SBIC dataset."""
    train = DataLoader(
        SBICDataset(config.SBIC_TRAIN_PATH, tokenizer),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )
    dev = DataLoader(
        SBICDataset(config.SBIC_DEV_PATH, tokenizer),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )
    test = DataLoader(
        SBICDataset(config.SBIC_TEST_PATH, tokenizer),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )
    return train, dev, test


def validate(
    dev_data: DataLoader,
    model: GPT2LMHeadModel,
    device: str,
    tokenizer: GPT2Tokenizer,
    is_test_set: bool = False,
) -> float:
    """Validate the performance on the dev dataset."""

    model.eval()

    total_loss = 0
    prediction_list, true_label_list = [], []
    with torch.no_grad():
        for batch in tqdm(dev_data):
            # unpack batch
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["input_ids"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

        predictions = torch.argmax(outputs.logits, dim=-1)
        for label, prediction in zip(labels, predictions):
            # TODO: should I use skip_special_tokens=False?
            true_text = tokenizer.decode(label, skip_special_tokens=True)
            pred_text = tokenizer.decode(prediction, skip_special_tokens=True)
            prediction_list.append(pred_text), true_label_list.append(true_text)

    # write output to file
    if is_test_set:
        write_output_to_csv(true_label_list, prediction_list, prefix="test")
    else:
        write_output_to_csv(true_label_list, prediction_list, prefix="val")

    # compute average validation loss
    avg_val_loss = total_loss / len(dev_data)

    # print and log the loss (test or validation set)
    loss_type = "Test" if is_test_set else "Validation"
    logging.info(f"{loss_type} Loss: {avg_val_loss}")
    print(f"{loss_type} Loss: {avg_val_loss}")

    model.train()

    return avg_val_loss


def train(
    train_data: DataLoader,
    dev_data: DataLoader,
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
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
    prediction_list, true_label_list = [], []
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
            )  # input is the first half of the sequence, label is the second half

            optim.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            optim.step()
            scheduler.step()

            total_loss += loss.item()

            # save predictions and true labels
            predictions = torch.argmax(outputs.logits, dim=-1)
            for label, prediction in zip(labels, predictions):
                # TODO: should I use skip_special_tokens=False?
                true_text = tokenizer.decode(label, skip_special_tokens=True)
                pred_text = tokenizer.decode(prediction, skip_special_tokens=True)
                prediction_list.append(pred_text), true_label_list.append(true_text)

        # compute average training loss for one epoch
        avg_train_loss = total_loss / len(train_data)
        logging.info(f"Epoch {idx+1} - Training Loss: {avg_train_loss}")
        print(f"Epoch {idx+1} - Training Loss: {avg_train_loss}")

        # write output to file
        write_output_to_csv(true_label_list, prediction_list, prefix="train")

        val_loss = validate(dev_data, model, device, tokenizer)

        # save current model version and val loss
        val_loss_history.append(val_loss)
        epoch_model_path = f"tmp/early_stopping/intermediate_model_epoch_{idx+1}.pt"
        torch.save(model.state_dict(), epoch_model_path)
        model_checkpoint_paths.append(epoch_model_path)

    # save only the best model
    best_epoch_idx = val_loss_history.index(min(val_loss_history))
    best_model_final_path = f"tmp/models/best_model_{epochs}_{timestamp}.pt"
    shutil.copy(model_checkpoint_paths[best_epoch_idx], best_model_final_path)
    logging.info(f"Best model epoch:{best_epoch_idx+1} at: {best_model_final_path}")
    logging.info(f"Best model val loss: {val_loss_history[best_epoch_idx]}")

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
    train_dataset, dev_dataset, test_dataset = load_data(tokenizer)

    print("Training model...")
    trained_model = train(train_dataset, dev_dataset, model, tokenizer)

    # check performance on the test set
    print("Validating model on the test set...")
    _ = validate(test_dataset, trained_model, get_device(), tokenizer, is_test_set=True)


if __name__ == "__main__":
    set_seed(config.SEED)
    main()
    logging.info(f"Training took {get_run_time(start)} min. Script Done.")
    print(f"Elapsed time: {get_run_time(start)} min.")
