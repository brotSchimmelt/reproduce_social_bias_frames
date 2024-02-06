import logging
import os
from datetime import datetime

from icecream import ic
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

import config
from utils.helper import get_run_time
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


def main():

    # load model and tokenizer
    model_path = config.GPT2_XL
    tokenizer = load_tokenizer(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))

    # load datasets
    train_data = SBICDataset(config.SBIC_TRAIN_PATH, tokenizer)
    dev_data = SBICDataset(config.SBIC_DEV_PATH, tokenizer)
    test_data = SBICDataset(config.SBIC_TEST_PATH, tokenizer)

    # early stopping is not in the paper
    # weight decay is also not in the paper
    training_args = TrainingArguments(
        output_dir="./tmp/results",
        num_train_epochs=config.EPOCHS,  # 1,2,5
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        warmup_steps=config.WARMUP_STEPS,  # number of warmup steps is not mentioned in the paper
        learning_rate=config.LEARNING_RATE,
        logging_dir="./logs",
        logging_steps=10,
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

    trainer.train()

    # final evaluation
    test_results = trainer.evaluate(test_data)
    print(f"Test Loss: {test_results['eval_loss']}")

    # save the model
    model_path = f"./best_model_{timestamp}"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)


if __name__ == "__main__":
    main()
    print(f"Elapsed time: {get_run_time(start)} min.")
