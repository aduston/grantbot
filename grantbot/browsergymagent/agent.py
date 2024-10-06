import os
from browsergym.experiments import Agent
from browsergym.experiments.agent import AgentInfo
from browsergym.utils.obs import flatten_axtree_to_str, prune_html
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from typing import Any

from llm_utils import DEFAULT_MODEL, ParseError, retry
import dynamic_prompting
from dynamic_prompting import Flags


class WebResearchAgent(Agent):
    """
    An agent that, given a starting URL and a goal, tries to
    complete the goal.
    """

    def obs_preprocessor(self, obs: dict) -> dict:
        return {
            "goal": obs["goal"],
            "axtree_txt": flatten_axtree_to_str(obs["axtree_object"]),
            "pruned_html": prune_html(obs["dom_txt"])
        }

    def __init__(self, goal: str, flags: Flags, 
                 model_name: str = DEFAULT_MODEL) -> None:
        super().__init__()
        self.goal = goal
        self.flags = flags
        self.model_name = model_name
        self.chat_llm = ChatOpenAI(
            model_name=DEFAULT_MODEL,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.1,
            max_tokens=2000,
        )
        self.action_set = dynamic_prompting.get_action_space(flags)
        self.obs_history = []
        self.actions = []
        self.memories = []
        self.thoughts = []

    def create_parser(
        self, main_prompt: dynamic_prompting.MainPrompt
    ) -> callable[[Any], tuple[dict, bool, str]]:
        def parser(text):
            try:
                ans_dict = main_prompt.parse_answer(text)
            except ParseError as e:
                # these parse errors will be caught by the retry function and
                # the chat_llm will have a chance to recover
                return None, False, str(e)

            return ans_dict, True, ""
        return parser

    def get_action(self, obs: Any) -> tuple[str, AgentInfo]:
        """
        For a description of `obs`, please see the get_action method
        in the superclass
        """
        self.obs_history.append(obs)
        main_prompt = dynamic_prompting.MainPrompt(
            self.obs_history,
            self.actions,
            self.memories,
            self.thoughts,
            self.flags
        )
        sys_msg = dynamic_prompting.SystemPrompt().prompt
        prompt = dynamic_prompting.fit_tokens(
            main_prompt,
            max_prompt_tokens=128000,
            model_name=self.model_name,
        )

        chat_messages = [
            SystemMessage(content=sys_msg),
            HumanMessage(content=prompt),
        ]

        parser = self.create_parser(main_prompt)
        try:
            ans_dict = retry(self.chat_llm, chat_messages, 4, parser)
        except ValueError as e:
            ans_dict = {
                "action": None,
                "err_msg": str(e),

            }


        return action, AgentInfo()
