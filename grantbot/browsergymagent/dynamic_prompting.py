import abc
import logging
import platform
from browsergym.core.action.highlevel import HighLevelActionSet
from llm_utils import DEFAULT_MODEL, count_tokens, image_to_jpg_base64_url, parse_html_tags_raise
from textwrap import dedent

logger = logging.getLogger(__name__)

class PromptElement:
    """Base class for all prompt elements. Prompt elements can be hidden.

    Prompt elements are used to build the prompt. Use flags to control which
    prompt elements are visible. We use class attributes as a convenient way
    to implement static prompts, but feel free to override them with instance
    attributes or @property decorator."""

    _prompt = ""
    _abstract_ex = ""
    _concrete_ex = ""

    def __init__(self, visible: bool = True) -> None:
        """Prompt element that can be hidden.

        Parameters
        ----------
        visible : bool, optional
            Whether the prompt element should be visible, by default True. Can
            be a callable that returns a bool. This is useful when a specific
            flag changes during a shrink iteration.
        """
        self._visible = visible

    @property
    def prompt(self):
        """Avoid overriding this method. Override _prompt instead."""
        return self._hide(self._prompt)

    @property
    def abstract_ex(self):
        """Useful when this prompt element is requesting an answer from the llm.
        Provide an abstract example of the answer here. See Memory for an
        example.

        Avoid overriding this method. Override _abstract_ex instead
        """
        return self._hide(self._abstract_ex)

    @property
    def concrete_ex(self):
        """Useful when this prompt element is requesting an answer from the llm.
        Provide a concrete example of the answer here. See Memory for an
        example.

        Avoid overriding this method. Override _concrete_ex instead
        """
        return self._hide(self._concrete_ex)

    @property
    def is_visible(self):
        """Handle the case where visible is a callable."""
        visible = self._visible
        if callable(visible):
            visible = visible()
        return visible

    def _hide(self, value):
        """Return value if visible is True, else return empty string."""
        if self.is_visible:
            return value
        else:
            return ""

    def _parse_answer(self, text_answer) -> dict:
        if self.is_visible:
            return self._parse_answer(text_answer)
        else:
            return {}
        
class Shrinkable(PromptElement, abc.ABC):
    @abc.abstractmethod
    def shrink(self) -> None:
        """Implement shrinking of this prompt element.

        You need to recursively call all shrinkable elements that are part of
        this prompt. You can also implement a shriking startegy for this prompt.
        Shrinking is can be called multiple times to progressively shrink the
        prompt until it fits max_tokens. Default max shrink iterations is 20.
        """
        pass

class Trunkater(Shrinkable):
    def __init__(self, visible, shrink_speed=0.3, start_trunkate_iteration=10):
        super().__init__(visible=visible)
        self.shrink_speed = shrink_speed
        self.start_trunkate_iteration = start_trunkate_iteration
        self.shrink_calls = 0
        self.deleted_lines = 0

    def shrink(self) -> None:
        if self.is_visible and self.shrink_calls >= self.start_trunkate_iteration:
            # remove the fraction of _prompt
            lines = self._prompt.splitlines()
            new_line_count = int(len(lines) * (1 - self.shrink_speed))
            self.deleted_lines += len(lines) - new_line_count
            self._prompt = "\n".join(lines[:new_line_count])
            self._prompt += f"\n... Deleted {self.deleted_lines} lines to reduce prompt size."

        self.shrink_calls += 1

def fit_tokens(
    shrinkable: Shrinkable, max_prompt_tokens=None, max_iterations=20, model_name=DEFAULT_MODEL
):
    """Shrink a prompt element until it fits max_tokens.

    Parameters
    ----------
    shrinkable : Shrinkable
        The prompt element to shrink.
    max_tokens : int
        The maximum number of tokens allowed.
    max_iterations : int, optional
        The maximum number of shrink iterations, by default 20.
    model_name : str, optional
        The name of the model used when tokenizing.

    Returns
    -------
    str : the prompt after shrinking.
    """

    if max_prompt_tokens is None:
        return shrinkable.prompt

    for _ in range(max_iterations):
        prompt = shrinkable.prompt
        if isinstance(prompt, str):
            prompt_str = prompt
        elif isinstance(prompt, list):
            prompt_str = "\n".join([p["text"] for p in prompt if p["type"] == "text"])
        else:
            raise ValueError(f"Unrecognized type for prompt: {type(prompt)}")
        n_token = count_tokens(prompt_str, model=model_name)
        if n_token <= max_prompt_tokens:
            return prompt
        shrinkable.shrink()

    logger.info(
        dedent(
            f"""\
            After {max_iterations} shrink iterations, the prompt is still
            {count_tokens(prompt_str)} tokens (greater than {max_prompt_tokens}). Returning the prompt as is."""
        )
    )
    return prompt

class HTML(Trunkater):
    def __init__(self, html, visible: bool = True, prefix="") -> None:
        super().__init__(visible=visible, start_trunkate_iteration=5)
        self._prompt = f"\n{prefix}HTML:\n{html}\n"


class AXTree(Trunkater):
    def __init__(self, ax_tree, visible: bool = True, coord_type=None, prefix="") -> None:
        super().__init__(visible=visible, start_trunkate_iteration=10)
        if coord_type == "center":
            coord_note = """\
Note: center coordinates are provided in parenthesis and are
  relative to the top left corner of the page.\n\n"""
        elif coord_type == "box":
            coord_note = """\
Note: bounding box of each object are provided in parenthesis and are
  relative to the top left corner of the page.\n\n"""
        else:
            coord_note = ""
        self._prompt = f"\n{prefix}AXTree:\n{coord_note}{ax_tree}\n"


class Error(PromptElement):
    def __init__(self, error, visible: bool = True, prefix="") -> None:
        super().__init__(visible=visible)
        self._prompt = f"\n{prefix}Error from previous action:\n{error}\n"

class Observation(Shrinkable):
    """Observation of the current step.

    Contains the html, the accessibility tree and the error logs.
    """

    def __init__(self, obs) -> None:
        self.obs = obs
        self.html = HTML(obs["pruned_html"], visible=True, prefix="## ")
        self.ax_tree = AXTree(
            obs["axtree_txt"],
            visible=False,
            coord_type="False",
            prefix="## ",
        )
        self.error = Error(
            obs["last_action_error"],
            visible=False and obs["last_action_error"],
            prefix="## ",
        )

    def shrink(self):
        self.ax_tree.shrink()
        self.html.shrink()

    @property
    def _prompt(self) -> str:
        return f"\n# Observation of current step:\n{self.html.prompt}{self.ax_tree.prompt}{self.error.prompt}\n\n"

    def add_screenshot(self, prompt):
        if isinstance(prompt, str):
            prompt = [{"type": "text", "text": prompt}]
        img_url = image_to_jpg_base64_url(self.obs["screenshot"])
        prompt.append({"type": "image_url", "image_url": {"url": img_url}})

        return prompt

class MacNote(PromptElement):
    def __init__(self) -> None:
        super().__init__(visible=platform.system() == "Darwin")
        self._prompt = (
            "\nNote: you are on mac so you should use Meta instead of Control for Control+C etc.\n"
        )


class BeCautious(PromptElement):
    def __init__(self, visible: bool = True) -> None:
        super().__init__(visible=visible)
        self._prompt = f"""\
\nBe very cautious. Avoid submitting anything before verifying the effect of your
actions. Take the time to explore the effect of safe actions first. For example
you can fill a few elements of a form, but don't click submit before verifying
that everything was filled correctly.\n"""


class GoalInstructions(PromptElement):
    def __init__(self, goal, visible: bool = True) -> None:
        super().__init__(visible)
        self._prompt = f"""\
# Instructions
Review the current state of the page and all other information to find the best
possible next action to accomplish your goal. Your answer will be interpreted
and executed by a program, make sure to follow the formatting instructions.

## Goal:
{goal}
"""


class ChatInstructions(PromptElement):
    def __init__(self, chat_messages, visible: bool = True) -> None:
        super().__init__(visible)
        self._prompt = f"""\
# Instructions

You are a UI Assistant, your goal is to help the user perform tasks using a web browser. You can
communicate with the user via a chat, in which the user gives you instructions and in which you
can send back messages. You have access to a web browser that both you and the user can see,
and with which only you can interact via specific commands.

Review the instructions from the user, the current state of the page and all other information
to find the best possible next action to accomplish your goal. Your answer will be interpreted
and executed by a program, make sure to follow the formatting instructions.

## Chat messages:

"""
        self._prompt += "\n".join(
            [
                f"""\
 - [{msg['role']}] {msg['message']}"""
                for msg in chat_messages
            ]
        )


class SystemPrompt(PromptElement):
    _prompt = """\
You are an agent trying to solve a web task based on the content of the page and
a user instructions. You can interact with the page and explore. Each time you
submit an action it will be sent to the browser and you will receive a new page."""


class Memory(PromptElement):
    _prompt = ""  # provided in the abstract and concrete examples

    _abstract_ex = """
<memory>
A list of things to remember for next steps.
</memory>
"""

    _concrete_ex = """
<memory>
I clicked on bid 32 to activate tab 2. The accessibility tree should mention
focusable for elements of the form at next step.
</memory>
"""

    def _parse_answer(self, text_answer):
        return parse_html_tags_raise(text_answer, optional_keys=["memory"], merge_multiple=True)


class Thought(PromptElement):
    _prompt = ""

    _abstract_ex = """
<think>
Step-by-step thoughts including, for example, calculations on coordinates. These 
thoughts describe the effect that your previous action had on the current content 
of the page.
</think>
"""
    _concrete_ex = """
<think>
My memory says that I filled the first name and last name, but I can't see any
content in the form. I need to explore different ways to fill the form. Perhaps
the form is not visible yet or some fields are disabled. I need to replan.
</think>
"""

    def _parse_answer(self, text_answer):
        return parse_html_tags_raise(text_answer, optional_keys=["think"], merge_multiple=True)