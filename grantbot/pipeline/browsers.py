import abc
from autogen_magentic_one.markdown_browser.requests_markdown_browser import RequestsMarkdownBrowser
from playwright.sync_api import sync_playwright
from markdownify import markdownify as md

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36 Edg/116.0.1938.81"


class Browser(abc.ABC):
    @abc.abstractmethod
    def obtain_markdown(self, link: str) -> str:
        pass


class RequestsBrowser(Browser):
    def __init__(self):
        self.browser = RequestsMarkdownBrowser(
            requests_get_kwargs={"timeout": 30, "headers": {"User-Agent": USER_AGENT}})

    def obtain_markdown(self, link: str) -> str:
        return self.browser.visit_page(link)


class PlaywrightBrowser(Browser):
    def obtain_markdown(self, link: str) -> str:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(user_agent=USER_AGENT)
            page = context.new_page()
            page.goto(link)
            html = page.content()
            browser.close()
        return md(html)


if __name__ == '__main__':
    link = "https://www.costco.com/charitable-giving.html"
    browser = RequestsBrowser()
    result = browser.obtain_markdown(link)
    print(result)
