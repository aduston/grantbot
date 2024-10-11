import os
from dataclasses import dataclass
from langchain_google_community import GoogleSearchAPIWrapper
from autogen_magentic_one.markdown_browser.requests_markdown_browser import RequestsMarkdownBrowser
from openai import OpenAI
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel
from typing import Tuple
from multiprocessing import Pool

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

@dataclass
class SearchResult:
    title: str
    link: str
    snippets: set[str]

@dataclass
class TokenCount:
    completion_tokens: int
    prompt_tokens: int

class AnswerWithQuote(BaseModel):
    answer: str
    quote: str | None

class AnswersWithQuotes(BaseModel):
    answers_with_quotes: list[AnswerWithQuote]

@dataclass
class ResearchLinkWithQuotesResult:
    answers_with_quotes: list[AnswerWithQuote]
    usage: CompletionUsage
    link: str


def obtain_search_results(grant_maker: str) -> list[SearchResult]:
    queries = [
        f"{grant_maker} grants",
        f"{grant_maker} grant eligibility criteria",
        f"{grant_maker} grant application procedure",
    ]
    all_results = {}
    for query in queries:
        query_results = GoogleSearchAPIWrapper().results(query, 5)
        for result_dict in query_results:
            link = result_dict["link"]
            if link not in all_results:
                all_results[link] = SearchResult(
                    title=result_dict["title"],
                    link=link,
                    snippets=set(result_dict["snippet"]),
                )
            else:
                all_results[link].snippets.add(result_dict["snippet"])
    return list(all_results.values())


def research_link(grant_maker: str, link: str, instruction: str) -> str:
    browser = RequestsMarkdownBrowser(viewport_size=1024 * 32)
    page_content = browser.visit_page(link)
    system_prompt = (
        "You are an expert web researcher who specializes in researching grants. "
        "A Google search has returned a web page. "
        "You will be given a task and the returned web page in markdown format. "
        "Given the page content alone, complete the task. Rely on the page content alone. "
        "Do not use any external resources. Reply in markdown. "
        "If there is something you can't find, just omit it from your response."
    )
    user_prompt = (
        f"<GOOGLE_SEARCH_QUERY>{grant_maker} grants</GOOGLE_SEARCH_QUERY>\n"
        f"<TASK>\n{instruction}\n</TASK>\n"
        f"<WEB_PAGE_CONTENT>\n{page_content}\n</WEB_PAGE_CONTENT>\n"
    )
    completion  = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": user_prompt},
        ],
    )
    return completion.choices[0].message.content

def research_link_with_quotes(grant_maker: str, link: str, instruction: str) -> ResearchLinkWithQuotesResult:
    browser = RequestsMarkdownBrowser(viewport_size=1024 * 32)
    page_content = browser.visit_page(link)
    system_prompt = (
        "You are an expert web researcher who specializes in researching grants. "
        "A Google search has returned a web page. "
        "You will be given a task and the returned web page in markdown format. "
        "Given the page content alone, answer the questions in the task. Rely on the page content alone. "
        "Do not use any external resources. "
        "Reply with a list of answers to the questions in the task, and try to accompany each fact with a "
        "verbatim quote from the page content that supports the answer. "
        "The response needs to consist of a json object int he following format:\n\n"
        '{"answers_with_quotes": ["answer": "answer1", "quote": "quote1"], ["answer": "answer2", "quote": "quote2"]}' 
        "\n\n"
        "If there's an answer you can't find a quote for, just make the quote null. "
        "If there's a question you can't answer, just omit it from your response."
    )
    user_prompt = (
        f"<GOOGLE_SEARCH_QUERY>{grant_maker} grants</GOOGLE_SEARCH_QUERY>\n"
        f"<TASK>\n{instruction}\n</TASK>\n"
        f"<WEB_PAGE_CONTENT>\n{page_content}\n</WEB_PAGE_CONTENT>\n"
    )
    completion = openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": user_prompt},
        ],
        response_format=AnswersWithQuotes
    )
    return ResearchLinkWithQuotesResult(
        completion.choices[0].message.parse.answers_with_quotes, 
        completion.usage,
        link
    )


def generate_grantmaker_research_report(grant_maker: str, instruction: str) -> str:
    """
    Generates a grantmaker research report in Markdown format.
    """
    search_results = obtain_search_results(grant_maker)
    with Pool(15) as p:
        research_responses = p.starmap(
            research_link_with_quotes, 
            [(grant_maker, result.link, instruction) for result in search_results]
        )
    total_completion_tokens = sum([r.usage.completion_tokens for r in research_responses])
    total_prompt_tokens = sum([r.usage.prompt_tokens for r in research_responses])
    system_prompt = (
        
    )