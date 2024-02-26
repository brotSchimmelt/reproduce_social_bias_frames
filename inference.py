import logging
import os
from datetime import datetime
from typing import List

from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

import config
from utils.helper import get_device, get_run_time

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
    batch_size: int = 2,
) -> List[str]:
    """Greedy decoding."""

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
        num_return_sequences=1,
        do_sample=False,
    )
    return [o[0]["generated_text"] for o in output]


def decode_sampling(
    generation_prompts: List[str],
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    batch_size: int = 2,
) -> List[str]:
    """Sample decoding."""

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
        num_return_sequences=10,
        do_sample=True,
    )

    model_output = []
    for o in output:
        unpacked_output = [item["generated_text"] for item in o]

        model_output.append(unpacked_output)

    return model_output


def main() -> None:
    device = get_device()
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH, padding_side="left")

    prompts = [
        f"{config.START_TOKEN} Fuc u say ? @AyyyeThatsChubb: All these hoes look good on IG <link> {config.SEP_TOKEN}",
        f"{config.START_TOKEN} Yo hoe is like a tick so she hunting for a blood {config.SEP_TOKEN}",
    ]

    # greedy decoding
    greedy_output = decode_greedy(prompts, model, tokenizer)
    logging.info(f"Greedy output: {greedy_output}")

    # sampling decoding
    sampling_output = decode_sampling(prompts, model, tokenizer)
    logging.info(f"Sampling output: {sampling_output}")


if __name__ == "__main__":
    MODEL_PATH = "/bigwork/nhwpknet/reproduce_social_bias_frames/tmp/models/gpt2-small_42_25_02-12_36_55"
    main()
    print(f"Elapsed time: {get_run_time(start)} min.")
    logging.info(f"Elapsed time: {get_run_time(start)} min.")
