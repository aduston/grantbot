import os
from dataclasses import dataclass
from langchain_google_community import GoogleSearchAPIWrapper
from autogen_magentic_one.markdown_browser.requests_markdown_browser import RequestsMarkdownBrowser
from openai import OpenAI
from openai.types.completion_usage import CompletionUsage
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel
from multiprocessing import Pool
from pyre_extensions import none_throws
from functools import cache
from typing_extensions import TypeAlias
import tiktoken
import logging
from browsers import RequestsBrowser, PlaywrightBrowser

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = "gpt-4o"

ChatMessage: TypeAlias = ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam

logger = logging.getLogger(__name__)


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
    search_result: SearchResult


@dataclass
class ResearchReport:
    report: str  # in markdown format
    token_count: TokenCount


@cache
def get_tokenizer(model_name=MODEL):
    return tiktoken.encoding_for_model(model_name.split("/")[-1])


def count_text_tokens(text, model=MODEL):
    enc = get_tokenizer(model)
    return len(enc.encode(text))


def count_message_tokens(messages: list[ChatMessage], model=MODEL):
    return sum(count_text_tokens(m["content"], model) for m in messages)


def obtain_search_results(grant_maker: str) -> list[SearchResult]:
    # hand-coded domain-specific query expansion
    queries = [
        f"{grant_maker} grants",
        f"{grant_maker} grant eligibility criteria",
        f"{grant_maker} grant application procedure",
        f"{grant_maker} recent grants education",
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


def research_link(search_result: SearchResult, messages: list[ChatMessage]) -> ResearchLinkWithQuotesResult:
    completion = openai_client.beta.chat.completions.parse(
        model=MODEL,
        messages=messages,
        response_format=AnswersWithQuotes
    )
    return ResearchLinkWithQuotesResult(
        none_throws(completion.choices[0].message.parsed).answers_with_quotes,
        none_throws(completion.usage),
        search_result
    )


def research_link_messages(
        grant_maker: str,
        link: str,
        instruction: str) -> list[ChatMessage]:
    logger.debug(f"Obtaining page content for {link}")
    browser = PlaywrightBrowser()
    page_content = browser.obtain_markdown(link)
    logger.debug(f"Page content for {link} is {len(page_content)} characters")
    system_prompt = (
        "You are an expert web researcher who specializes in researching grants. "
        "A Google search has returned a web page. "
        "You will be given a task and the returned web page in markdown format. "
        "Given the page content alone, answer the questions in the task. Rely on the page content alone. "
        "Do not use any external resources. "
        "Reply with a list of answers to the questions in the task, and try to accompany each fact with a "
        "verbatim quote from the page content that supports the answer. "
        "The response needs to consist of a json object in the following format:\n\n"
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
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def research_links_messages(
        grant_maker: str,
        links: list[str],
        instruction: str) -> dict[str, list[ChatMessage]]:
    with Pool(15) as p:
        message_lists = p.starmap(
            research_link_messages,
            [(grant_maker, link, instruction) for link in links]
        )
    return dict(zip(links, message_lists))


def generate_research_responses(grant_maker: str, instruction: str) -> list[ResearchLinkWithQuotesResult]:
    search_results = obtain_search_results(grant_maker)
    # maybe in future actually process PDFs. For now, remove them.
    search_results = [r for r in search_results if not r.link.endswith(".pdf")]
    search_results_dict = {r.link: r for r in search_results}
    messages_dict = research_links_messages(
        grant_maker, list(search_results_dict.keys()), instruction)
    search_result_messages = [(search_results_dict[link], messages)
                              for link, messages in messages_dict.items()]
    with Pool(15) as p:
        research_responses = p.starmap(
            research_link,
            search_result_messages
        )
    return research_responses


def generate_grantmaker_research_report(grant_maker: str, instruction: str) -> ResearchReport:
    """
    Generates a grantmaker research report in Markdown format.
    """
    research_responses = generate_research_responses(grant_maker, instruction)
    system_prompt = (
        "You are an expert web researcher who specializes in researching grants. "
        "Given a task and a list of web pages, you have looked at each page individually "
        "and used it to answer questions in the task. When possible, you quoted page "
        "content to support your answer. You must now synthesize the "
        "individual answers into a report in Markdown format that addresses all "
        "questions in the task.\n\n"
        "When possible, add footnotes to your Markdown, quoting sources. Use the "
        "Markdown footnote syntax. "
        "Each footnote should include a link to the source page and the text of "
        "the quote that supports the information."
    )
    user_prompt = (
        f"<TASK>\n{instruction}\n</TASK>\n"
        f"<YOUR_ANSWERS>\n{research_results_prompt(
            research_responses)}\n</YOUR_ANSWERS>\n"
    )
    completion = openai_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    report = none_throws(completion.choices[0].message.content)
    usage = none_throws(completion.usage)
    total_completion_tokens = sum(
        [r.usage.completion_tokens for r in research_responses]) + \
        usage.completion_tokens
    total_prompt_tokens = sum(
        [r.usage.prompt_tokens for r in research_responses]) + \
        usage.prompt_tokens
    return ResearchReport(
        report=report,
        token_count=TokenCount(
            total_completion_tokens,
            total_prompt_tokens
        )
    )


def research_results_prompt(result: list[ResearchLinkWithQuotesResult]) -> str:
    prompt_lines = []
    for r in result:
        prompt_lines.append(
            f"# Answers from {r.search_result.link} (\"{r.search_result.title}\")")
        for a in r.answers_with_quotes:
            quote = '(none)' if a.quote is None else f'"{a.quote}"'
            prompt_lines.append(f"**Answer:** {a.answer}")
            prompt_lines.append(f"**Quote:** {quote}")
    return "\n\n".join(prompt_lines)


if __name__ == '__main__':
    from langchain.prompts import PromptTemplate
    from sample_data import MAIN_INSTRUCTION, RWF_PROGRAM_NAME, RWF_SUMMARY
    from pprint import pprint

    logger.info("Starting")

    grant_maker = "The Morris and Gwendolyn Cafritz Foundation"
    # grant_maker = "Costco"
    instruction = PromptTemplate.from_template(MAIN_INSTRUCTION).format(
        program_summary=RWF_SUMMARY, program_name=RWF_PROGRAM_NAME,
        grant_maker=grant_maker
    )
    research_report = generate_grantmaker_research_report(
        grant_maker, instruction)
    print(research_report.report)
