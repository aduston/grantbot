from browsergym.experiments import Agent
from browsergym.experiments.agent import AgentInfo
from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.utils.obs import flatten_axtree_to_str
from openai import OpenAI
from typing import Any

from llm_utils import DEFAULT_MODEL


class WebResearchAgent(Agent):
    """
    An agent that, given a starting URL, tries to answer
    questions
    """

    action_set = HighLevelActionSet(
        subsets=["chat", "bid"],  # define a subset of the action space
        # allow the agent to also use x,y coordinates
        # subsets=["chat", "bid", "coord"]
        strict=False,  # less strict on the parsing of the actions
        multiaction=True,  # enable to agent to take multiple actions at once
        demo_mode="default",  # add visual effects
    )

    def obs_preprocessor(self, obs: dict) -> dict:
        return {
            "goal": obs["goal"],
            "axtree_txt": flatten_axtree_to_str(obs["axtree_object"]),
        }

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        super().__init__()
        self.model_name = model_name
        self.openai_client = OpenAI()

    def get_action(self, obs: Any) -> tuple[str, AgentInfo]:
        """
        For a description of `obs`, please see the get_action method
        in the superclass
        """
        system_msg = f"""\
# Instructions
Review the current state of the page and all other information to find the best
possible next action to accomplish your goal. Your answer will be interpreted
and executed by a program, make sure to follow the formatting instructions.

# Goal:
{obs["goal"]}"""

        prompt = f"""\
# Current Accessibility Tree:
{obs["axtree_txt"]}

# Action Space
{self.action_set.describe(with_long_description=False, with_examples=True)}

Here is an example with chain of thought of a valid action when clicking on a
button:
"
In order to accomplish my goal I need to click on the button with bid 12
```click("12")```
"
"""

        # query OpenAI model
        response = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
        )
        action = response.choices[0].message.content

        return action, AgentInfo()
