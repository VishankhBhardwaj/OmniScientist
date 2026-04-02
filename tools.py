from langchain_core.tools import tool
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
from langchain_experimental.utilities import PythonREPL
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
import os

os.environ["WOLFRAM_ALPHA_APPID"] = ""
@tool
def python_commands(code:str)->str:
    """Execute python code and return the output"""
    repl = PythonREPL()
    return repl.run(code)

@tool
def arxiv_search(query:str)->str:
    """Search Arxiv for papers matching the query"""
    arxiv = ArxivQueryRun()
    return arxiv.run(query)

@tool
def wikipedia_search(query:str)->str:
    """Search Wikipedia for information matching the query"""
    wikipedia = WikipediaQueryRun()
    return wikipedia.run(query)

@tool
def wolfram_alpha(query:str)->str:
    """Search Wolfram Alpha for information matching the query"""
    wolfram = WolframAlphaAPIWrapper()
    return wolfram.run(query)
@tool
def web_search(query: str) -> str:
    """Search the internet for accurate and up to date information"""
    tavily = TavilySearchResults(max_results=3)
    return tavily.run(query)