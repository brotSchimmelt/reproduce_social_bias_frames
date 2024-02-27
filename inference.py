import argparse
import logging
import os
from datetime import datetime
from typing import List, Tuple

import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline, set_seed

import config
from utils.helper import get_device, get_run_time

#### ARGPARSE SETUP ####
parser = argparse.ArgumentParser(
    description="Run text generation with specified model."
)
parser.add_argument(
    "--model_name", type=str, help="Name of the model to use (without default value)."
)
# Add the new argument for max_length
parser.add_argument(
    "--max_length",
    type=int,
    default=config.MAX_LENGTH,
    help="Max length context window size for the LLM.",
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
        max_length=args.max_length,
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
        max_length=args.max_length,
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
    for i, outputs in enumerate(zip(*sampling_outputs)):  # transpose the list of lists
        col_name = f"sampling_output_{i+1}"
        df[col_name] = outputs


def choose_best_sampling_output(
    sampling_outputs: List[List[str]], model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer
) -> Tuple[List[str], List[int], List[float]]:
    """Choose the best sampling output based on the highest likelihood."""

    best_output, best_idxs, best_scores = [], [], []
    for output_list in sampling_outputs:
        likelihoods = []
        for candidate in output_list:
            tokens = tokenizer.encode(candidate, return_tensors="pt")

            with torch.no_grad():
                outputs = model(tokens, labels=tokens)
                # the model returns the loss, which is the negative log likelihood of the tokens
                # since we're using the same tokens as both input and labels, this gives us the
                # negative log likelihood of the entire sequence
                negative_log_likelihood = outputs.loss
                likelihoods.append(-negative_log_likelihood)

        # find the index of the highest likelihood
        best_idx = likelihoods.index(max(likelihoods))

        # save the best output, its index and the likelihood#
        best_output.append(output_list[best_idx])
        best_idxs.append(best_idx)
        best_scores.append(likelihoods[best_idx])

    return best_output, best_idxs, best_scores


def main() -> None:
    # load model and tokenizer
    device = get_device()
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(
        MODEL_PATH, padding_side=config.PADDING_SIDE
    )
    default_tok = GPT2Tokenizer.from_pretrained(MODEL_PATH)

    # load prompts
    df_dev = pd.read_csv(config.DEV_EVAL_PROMPTS)
    df_test = pd.read_csv(config.TEST_EVAL_PROMPTS)
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

    # choose the best sampling output
    logging.info("Choosing the best sampling output based on the highest likelihood.")
    test_best_sampling_output = choose_best_sampling_output(
        test_sampling_output, model, default_tok
    )
    logging.info("Best sampling output for test was chosen.")
    dev_best_sampling_output = choose_best_sampling_output(
        dev_sampling_output, model, default_tok
    )
    logging.info("Best sampling output for dev was chosen.")

    # save results
    df_dev["greedy_output"] = dev_greedy_output
    add_sampling_outputs_to_df(df_dev, dev_sampling_output)
    df_dev["best_sampling_output"] = dev_best_sampling_output[0]
    df_dev["best_sampling_output_idx"] = dev_best_sampling_output[1]
    df_dev["best_sampling_output_score"] = dev_best_sampling_output[2]

    df_test["greedy_output"] = test_greedy_output
    add_sampling_outputs_to_df(df_test, test_sampling_output)
    df_test["best_sampling_output"] = test_best_sampling_output[0]
    df_test["best_sampling_output_idx"] = test_best_sampling_output[1]
    df_test["best_sampling_output_score"] = test_best_sampling_output[2]

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    df_dev.to_csv(OUTPUT_PATH + "output_dev.csv", index=False)
    df_test.to_csv(OUTPUT_PATH + "output_test.csv", index=False)


if __name__ == "__main__":
    main()
    print(f"Elapsed time: {get_run_time(start)} min.")
    logging.info(f"Elapsed time: {get_run_time(start)} min.")
