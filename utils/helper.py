import hashlib
import html
import logging
import re
from datetime import datetime
from typing import List

import pandas as pd
import torch
from icecream import ic
from transformers import TrainerCallback

import config


class EvalLoggingCallback(TrainerCallback):
    """Custom callback for logging evaluation results separately."""

    def __init__(self, *args, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            if k == "output_dir":
                setattr(self, k, v)

    def on_evaluate(self, args, state, control, **kwargs):
        # Log validation loss separately
        with open(self.output_dir, "a") as log_file:
            log_file.write(f"{state.global_step},{kwargs['metrics']['eval_loss']}\n")


def get_device() -> str:
    """
    Determines the most suitable device for PyTorch operations.

    Returns:
    - str: The identifier of the most suitable device ("cuda", "mps", or "cpu").
    """
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def get_run_time(start_time: datetime) -> str:
    """
    Calculate and format the elapsed time between the given start time
    and the current time.

    Parameters:
        start_time (datetime): The starting time from which the elapsed
        time is calculated.

    Returns:
        str: A formatted string representing the elapsed time in 'MM:SS'
        format, with leading zeros for minutes and seconds.
    """
    elapsed_time = (datetime.now() - start_time).total_seconds()

    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    return f"{minutes:02}:{seconds:02}"


def clean_post(post: str) -> str:
    """
    Clean and preprocess a social media post.

    Args:
        post (str): The input social media post.

    Returns:
        str: A cleaned and processed version of the input post.
    """
    if not isinstance(post, str):
        ic(type(post))
        logging.info(f"Input post is not a string: {post} but {type(post)}")

    # replace certain characters
    post = post.replace("&amp;", "&")
    post = post.replace("\n", " ")
    post = html.unescape(post)
    post = replace_links(post)

    # remove certain characters
    post = remove_multiple_whitespace(post)
    chars_to_remove = ["`", '"', "“", "”", "\u200f", "*", "_", "-"]
    for c in chars_to_remove:
        post = post.replace(c, "")

    return post.strip()


def remove_multiple_whitespace(s: str) -> str:
    """
    Remove multiple consecutive whitespace characters (including spaces,
    tabs, and newlines) from the input string and replace them with a single space.

    Args:
        s (str): The input string containing multiple whitespace characters.

    Returns:
        str: A new string with multiple consecutive whitespace characters replaced
        by a single space.
    """
    return re.sub(r"\s+", " ", s)


def replace_links(s: str) -> str:
    """
    Replace links (http or https) and Twitter links from a given string with '<link>'.

    Args:
        s (str): The input string containing text that may include links.

    Returns:
        str: A modified string with links replaced by '<link>'.
    """
    link_pattern = re.compile(
        r"(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)|(pic\.twitter\.com/\S+)"
    )
    return link_pattern.sub("<link>", s).strip()


def write_output_to_csv(
    true_labels: List[str], predictions: List[str], prefix: str = "val"
) -> None:
    """
    Writes the true labels and predictions to a CSV file in the tmp/ directory.
    The file is named with the current timestamp to ensure uniqueness.

    Args:
    - true_labels (List[str]): A list of true labels.
    - predictions (List[str]): A list of predicted labels.
    - prefix (str, optional): A prefix to use in the filename. Defaults to "val".
    """
    true_labels = [label.replace(config.PAD_TOKEN, "") for label in true_labels]
    predictions = [pred.replace(config.PAD_TOKEN, "") for pred in predictions]
    df = pd.DataFrame({"true_labels": true_labels, "predictions": predictions})
    now = datetime.now().strftime("%d-%m_%H-%M-%S")
    df.to_csv(f"tmp/output/{prefix}_{now}.csv", index=False)


def create_md5_hash(input_string: str) -> str:
    """
    Generates an MD5 hash for the given input string.

    Args:
    - input_string (str): The string to be hashed.

    Returns:
    - str: The hexadecimal MD5 hash of the input string.
    """
    hash_object = hashlib.md5(input_string.encode())
    return hash_object.hexdigest()
