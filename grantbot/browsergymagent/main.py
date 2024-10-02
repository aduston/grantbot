import gymnasium as gym
import browsergym.core


def main() -> None:
    env = gym.make(
        "browsergym/openended",
        task_kwargs={"start_url": "https://www.google.com/"},  # starting URL
        # wait for a user message after each agent message sent to the chat
        wait_for_user_message=True,
    )
    obs, info = env.reset()
    done = False
    while not done:
        action = None
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated