import gymnasium as gym
from langchain.prompts import PromptTemplate

from sample_data import MAIN_INSTRUCTION, RWF_PROGRAM_NAME, RWF_SUMMARY
from browsergymagent.agent import WebResearchAgent
from browsergymagent.dynamic_prompting import Flags


def browsergymagent_main(start_url: str, agent_instruction: str) -> None:
    env = gym.make(
        "browsergym/openended",
        task_kwargs={"start_url": start_url, "goal": agent_instruction},
        # wait for a user message after each agent message sent to the chat
        wait_for_user_message=False,
        headless=False,
    )
    obs, _info = env.reset()
    done = False
    agent = WebResearchAgent(agent_instruction, Flags())
    while not done:
        action = agent.get_action(agent.obs_preprocessor(obs))
        obs, _reward, terminated, truncated, _info = env.step(action)
        done = terminated or truncated


if __name__ == '__main__':
    instruction = PromptTemplate.from_template(MAIN_INSTRUCTION).format(
        program_summary=RWF_SUMMARY, program_name=RWF_PROGRAM_NAME,
        grant_maker="The Morris and Gwendolyn Cafritz Foundation"
    )
    browsergymagent_main(
        "https://www.cafritzfoundation.org/",
        instruction,
    )
