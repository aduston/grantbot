"""
A bunch of utility functions for dealing with LLMs
"""

import base64
import io
import logging
import re
import time
from functools import cache
from typing import Any, Callable

import numpy as np
import tiktoken
from PIL import Image
from openai import RateLimitError
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.base import BaseMessage
from langchain_core.messages.human import HumanMessage

DEFAULT_MODEL = "gpt-4o"

logger = logging.getLogger(__name__)


def _extract_wait_time(error_message, min_retry_wait_time=60):
    """Extract the wait time from an OpenAI RateLimitError message."""
    match = re.search(r"try again in (\d+(\.\d+)?)s", error_message)
    if match:
        return max(min_retry_wait_time, float(match.group(1)))
    return min_retry_wait_time


def retry(
    chat: BaseChatModel,
    messages: list[BaseMessage],
    n_retry: int,
    parser: Callable[[Any], tuple[dict, bool, str]],
    log: bool = True,
    min_retry_wait_time=60,
    rate_limit_max_wait_time=60 * 30,
):
    """Retry querying the chat models with the response from the parser until
    it returns a valid value.

    If the answer is not valid, it will retry and append to the chat the retry
    message.  It will stop after `n_retry`.

    Note, each retry has to resend the whole prompt to the API. This can be
    slow and expensive.

    Parameters:
    -----------
        chat (function) : a langchain ChatOpenAI taking a list of messages and
            returning a list of answers.
        messages (list) : the list of messages so far.
        n_retry (int) : the maximum number of sequential retries.
        parser (function): a function taking a message and returning a tuple
        with the following fields:
            value : the parsed value,
            valid : a boolean indicating if the value is valid,
            retry_message : a message to send to the chat if the value is not
                valid
        log (bool): whether to log the retry messages.
        min_retry_wait_time (float): the minimum wait time in seconds
            after RateLimtError. will try to parse the wait time from the error
            message.

    Returns:
    --------
        value: the parsed value
    """
    tries = 0
    rate_limit_total_delay = 0
    while tries < n_retry and \
            rate_limit_total_delay < rate_limit_max_wait_time:
        try:
            answer = chat.invoke(messages)
        except RateLimitError as e:
            wait_time = _extract_wait_time(e.args[0], min_retry_wait_time)
            logger.warning(
                "RateLimitError, waiting %s before retrying.", wait_time
            )
            time.sleep(wait_time)
            rate_limit_total_delay += wait_time
            if rate_limit_total_delay >= rate_limit_max_wait_time:
                logger.warning(
                    "Total wait time for rate limit exceeded. "
                    "Waited %ds > %ds.",
                    rate_limit_total_delay, rate_limit_max_wait_time
                )
                raise
            continue

        messages.append(answer)

        value, valid, retry_message = parser(answer.content)
        if valid:
            return value

        tries += 1
        if log:
            logging.info(
                "Query failed. Retrying %d/%d.\n[LLM]:\n%s\n[User]:\n%s",
                tries, n_retry, answer.content, retry_message
            )
        messages.append(HumanMessage(content=retry_message))

    raise ValueError(f"Could not parse a valid value after {n_retry} retries.")


@cache
def get_tokenizer(model_name=DEFAULT_MODEL):
    return tiktoken.encoding_for_model(model_name.split("/")[-1])


def count_tokens(text, model=DEFAULT_MODEL):
    enc = get_tokenizer(model)
    return len(enc.encode(text))


def image_to_jpg_base64_url(image: np.ndarray | Image.Image):
    """Convert a numpy array to a base64 encoded image url."""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")

    image_base64 = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{image_base64}"


def extract_html_tags(text, keys):
    """Extract the content within HTML tags for a list of keys.

    Parameters
    ----------
    text : str
        The input string containing the HTML tags.
    keys : list of str
        The HTML tags to extract the content from.

    Returns
    -------
    dict
        A dictionary mapping each key to a list of subset in `text`
        that match the key.

    Notes
    -----
    All text and keys will be converted to lowercase before matching.

    """
    content_dict = {}
    # text = text.lower()
    # keys = set([k.lower() for k in keys])
    for key in keys:
        pattern = f"<{key}>(.*?)</{key}>"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            content_dict[key] = [match.strip() for match in matches]
    return content_dict


class ParseError(Exception):
    pass


def parse_html_tags_raise(
    text, keys=(), optional_keys=(), merge_multiple=False
):
    """
    A version of parse_html_tags that raises an exception if the parsing is 
    not successful.
    """
    content_dict, valid, retry_message = parse_html_tags(
        text, keys, optional_keys, merge_multiple=merge_multiple
    )
    if not valid:
        raise ParseError(retry_message)
    return content_dict


def parse_html_tags(text, keys=(), optional_keys=(), merge_multiple=False):
    """
    Satisfy the parse api, extracts 1 match per key and validates that
    all keys are present

    Parameters
    ----------
    text : str
        The input string containing the HTML tags.
    keys : list of str
        The HTML tags to extract the content from.
    optional_keys : list of str
        The HTML tags to extract the content from, but are optional.

    Returns
    -------
    dict
        A dictionary mapping each key to subset of `text` that match the key.
    bool
        Whether the parsing was successful.
    str
        A message to be displayed to the agent if the parsing was not
        successful.
    """
    all_keys = tuple(keys) + tuple(optional_keys)
    content_dict = extract_html_tags(text, all_keys)
    retry_messages = []

    for key in all_keys:
        if key not in content_dict:
            if key not in optional_keys:
                retry_messages.append(
                    f"Missing the key <{key}> in the answer."
                )
        else:
            val = content_dict[key]
            content_dict[key] = val[0]
            if len(val) > 1:
                if not merge_multiple:
                    retry_messages.append(
                        f"Found multiple instances of the key {key}. "
                        "You should have only one of them."
                    )
                else:
                    # merge the multiple instances
                    content_dict[key] = "\n".join(val)

    valid = len(retry_messages) == 0
    retry_message = "\n".join(retry_messages)
    return content_dict, valid, retry_message
