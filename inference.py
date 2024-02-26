import argparse
import logging
import os
from datetime import datetime
from typing import List

import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline, set_seed

import config
from utils.helper import get_device, get_run_time

#### ARGPARSE SETUP ####
parser = argparse.ArgumentParser(
    description="Run text generation with specified model."
)
parser.add_argument(
    "model_name", type=str, help="Name of the model to use (without default value)."
)
args = parser.parse_args()

MODEL_PATH = config.MODEL_PATH + args.model_name
OUTPUT_PATH = config.OUTPUT_PATH + args.model_name + "/"
#### ARGPARSE SETUP ####

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


def decode_greedy(
    generation_prompts: List[str],
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    batch_size: int = config.INFERENCE_BATCH_SIZE_GREEDY,
) -> List[str]:
    """Greedy decoding."""

    set_seed(config.DEFAULT_SEED)

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0,
        batch_size=batch_size,
    )

    output = generator(
        generation_prompts,
        max_length=config.MAX_LENGTH,
        truncation=True,
        num_return_sequences=1,
        do_sample=False,
        pad_token_id=generator.tokenizer.eos_token_id,
    )
    return [o[0]["generated_text"].strip() for o in output]


def decode_sampling(
    generation_prompts: List[str],
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    batch_size: int = config.INFERENCE_BATCH_SIZE_SAMPLING,
) -> List[List[str]]:
    """Sampling based decoding."""

    set_seed(config.DEFAULT_SEED)

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0,
        batch_size=batch_size,
    )

    output = generator(
        generation_prompts,
        max_length=config.MAX_LENGTH,
        truncation=True,
        num_return_sequences=config.NUM_RETURN_SEQ,
        do_sample=True,
        pad_token_id=generator.tokenizer.eos_token_id,
    )

    return [[item["generated_text"].strip() for item in o] for o in output]


def add_sampling_outputs_to_df(
    df: pd.DataFrame, sampling_outputs: List[List[str]]
) -> None:
    """
    Adds each list in sampling_outputs as a new column in the given DataFrame.
    """
    # for i, output in enumerate(sampling_outputs):
    #     col_name = f"sampling_output_{i+1}"
    #     df[col_name] = output
    for i, outputs in enumerate(zip(*sampling_outputs)):  # transpose the list of lists
        col_name = f"sampling_output_{i+1}"
        df[col_name] = outputs


def main() -> None:
    # load model and tokenizer
    device = get_device()
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(
        MODEL_PATH, padding_side=config.PADDING_SIDE
    )

    # load prompts
    df_dev = pd.read_csv(config.DATA_PATH + "dev_eval_prompts.csv")
    df_test = pd.read_csv(config.DATA_PATH + "test_eval_prompts.csv")
    dev_prompts = df_dev["prompt"].tolist()
    test_prompts = df_test["prompt"].tolist()
    logging.info(
        f"Loaded {len(dev_prompts)} dev prompts and {len(test_prompts)} test prompts."
    )
    logging.info(f"First dev prompt: {dev_prompts[0]}")
    logging.info(f"First test prompt: {test_prompts[0]}")

    # greedy decoding
    dev_greedy_output = decode_greedy(dev_prompts, model, tokenizer)
    test_greedy_output = decode_greedy(test_prompts, model, tokenizer)
    logging.info(f"Greedy output for dev: {dev_greedy_output[0]}")
    logging.info(f"Greedy output for test: {test_greedy_output[0]}")

    # sampling decoding
    dev_sampling_output = decode_sampling(dev_prompts, model, tokenizer)
    test_sampling_output = decode_sampling(test_prompts, model, tokenizer)
    logging.info(f"Sampling output for dev: {dev_sampling_output[0][0]}")
    logging.info(f"Sampling output for test: {test_sampling_output[0][0]}")

    # save results
    df_dev["greedy_output"] = dev_greedy_output
    df_test["greedy_output"] = test_greedy_output
    add_sampling_outputs_to_df(df_dev, dev_sampling_output)
    add_sampling_outputs_to_df(df_test, test_sampling_output)

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    df_dev.to_csv(OUTPUT_PATH + "output_dev.csv", index=False)
    df_test.to_csv(OUTPUT_PATH + "output_test.csv", index=False)


if __name__ == "__main__":
    main()
    print(f"Elapsed time: {get_run_time(start)} min.")
    logging.info(f"Elapsed time: {get_run_time(start)} min.")
