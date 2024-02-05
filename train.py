import logging
import os
from datetime import datetime

from icecream import ic
from transformers import GPT2LMHeadModel, GPT2Tokenizer

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


def main() -> None:

    # load model and tokenizer
    model_path = config.GPT2_SMALL
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

    # load model
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))  # since we added new tokens

    # load datasets
    train_dataset = SBICDataset(config.SBIC_TRAIN_PATH, tokenizer)
    dev_dataset = SBICDataset(config.SBIC_DEV_PATH, tokenizer)
    ic(len(train_dataset), len(dev_dataset))


if __name__ == "__main__":
    main()
    print(f"Elapsed time: {get_run_time(start)} min.")
