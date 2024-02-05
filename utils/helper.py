import html
import logging
import re
from datetime import datetime

from icecream import ic


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
