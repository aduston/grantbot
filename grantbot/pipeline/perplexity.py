import os
import requests
from shared_types import GrantInformation
from openai import OpenAI
from langchain.prompts import PromptTemplate

from sample_data import MAIN_INSTRUCTION, RWF_PROGRAM_NAME, RWF_SUMMARY

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def obtain_grant_info(prompt: str) -> GrantInformation | None:
    perplexity_response = obtain_perplexity_response(prompt)
    prompt = (
        "# The Researcher's Task\n"
        "We asked a web research agent to perform the following task:\n\n"
        f"{prompt}"
        "\n\n"
        "# The Researcher's Response\n"
        "The research agent's response was:\n\n"
        f"{perplexity_response}"
        "# Your task\n"
        "Your task is to extract the agent's response into a structured "
        "format."
    )
    completion = openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract the agent's response into a structured format."
                )
            },
            {"role": "user", "content": prompt},
        ],
        response_format=GrantInformation
    )
    return completion.choices[0].message.parsed


def obtain_perplexity_response(
        prompt: str,
        model_name: str = "llama-3.1-sonar-huge-128k-online") -> str:
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content":
                    "Be complete. Be concise. Be clear. Reply in Markdown."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "return_citations": True,
        "return_images": False,
        "return_related_questions": False,
        "search_recency_filter": "month",
        "top_k": 0,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 1
    }
    headers = {
        "Authorization": "Bearer " + os.environ["PERPLEXITY_API_KEY"],
        "Content-Type": "application/json"
    }
    response = requests.post(
        "https://api.perplexity.ai/chat/completions",
        json=payload,
        headers=headers,
        timeout=120
    )
    return response.text


if __name__ == "__main__":
    instruction = PromptTemplate.from_template(MAIN_INSTRUCTION).format(
        program_summary=RWF_SUMMARY, program_name=RWF_PROGRAM_NAME,
        grant_maker="The Morris and Gwendolyn Cafritz Foundation"
    )
    grant_info = obtain_grant_info(instruction)
    print(grant_info)
