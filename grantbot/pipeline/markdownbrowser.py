import os
from dataclasses import dataclass
from langchain_google_community import GoogleSearchAPIWrapper
from autogen_magentic_one.markdown_browser.requests_markdown_browser import RequestsMarkdownBrowser
from openai import OpenAI
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel
from multiprocessing import Pool
from pyre_extensions import none_throws

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
    search_result: SearchResult

@dataclass
class ResearchReport:
    report: str # in markdown format
    token_count: TokenCount

def obtain_search_results(grant_maker: str) -> list[SearchResult]:
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
    return none_throws(completion.choices[0].message.content)

def research_link_with_quotes(grant_maker: str, search_result: SearchResult, instruction: str) -> ResearchLinkWithQuotesResult:
    browser = RequestsMarkdownBrowser(viewport_size=1024 * 32)
    page_content = browser.visit_page(search_result.link)
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
    completion = openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": user_prompt},
        ],
        response_format=AnswersWithQuotes
    )
    return ResearchLinkWithQuotesResult(
        none_throws(completion.choices[0].message.parsed).answers_with_quotes, 
        none_throws(completion.usage),
        search_result
    )

def research_link_with_quotes_prompt(
        grant_maker: str, 
        search_result: SearchResult, 
        instruction: str) -> list[ChatCompletionMessageParam]:
    browser = RequestsMarkdownBrowser(viewport_size=1024 * 32)
    page_content = browser.visit_page(search_result.link)
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
        { "role": "system", "content": system_prompt },
        { "role": "user", "content": user_prompt},
    ]


def generate_grantmaker_research_report(grant_maker: str, instruction: str) -> ResearchReport:
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
        f"<YOUR_ANSWERS>\n{research_results_prompt(research_responses)}\n</YOUR_ANSWERS>\n"
    )
    completion  = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": user_prompt},
        ],
    )
    report = none_throws(completion.choices[0].message.content)
    usage = none_throws(completion.usage)
    return ResearchReport(
        report=report, 
        token_count=TokenCount(
            total_completion_tokens + usage.completion_tokens, 
            total_prompt_tokens + usage.prompt_tokens
        )
    )


def research_results_prompt(result: list[ResearchLinkWithQuotesResult]) -> str:
    prompt_lines = []
    for r in result:
        prompt_lines.append(f"# Answers from {r.link}")
        for a in r.answers_with_quotes:
            quote = '(none)' if a.quote is None else f'"{a.quote}"'
            prompt_lines.append(f"**Answer:** {a.answer}")
            prompt_lines.append(f"**Quote:** {quote}")
    return "\n\n".join(prompt_lines)

if __name__ == '__main__':
    from langchain.prompts import PromptTemplate
    from sample_data import MAIN_INSTRUCTION, RWF_PROGRAM_NAME, RWF_SUMMARY

    # grant_maker = "The Morris and Gwendolyn Cafritz Foundation"
    grant_maker = "Costco"
    instruction = PromptTemplate.from_template(MAIN_INSTRUCTION).format(
        program_summary=RWF_SUMMARY, program_name=RWF_PROGRAM_NAME,
        grant_maker="The Morris and Gwendolyn Cafritz Foundation"
    )
    report = generate_grantmaker_research_report(grant_maker, instruction)
    print(report)