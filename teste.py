from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
import os
from dotenv import load_dotenv

# 1 - configuracoes iniciais
load_dotenv()
API_KEY = os.getenv("API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
model = ChatOpenAI(
    model="o4-mini",
    api_key=API_KEY
)

# 2 - Prompt do Sistema
system_message = SystemMessage(content="""
Você é um pesquisador muito sarcástico e irônico.
Use ferramenta 'search' sempre que necessário, especialmente para perguntas que exigem informaçõesa da web
"""
)

# 3 - criando a ferramenta search
@tool("search")
def search_web(query: str = "") -> str:
    """
    busca informacoes na web baseada na consulta fornecida.

    Args:
        query: Termos para buscar dados na web

    Returns:
        as informacoes encontradas na web ou uma mensagem indicando que nenhuma informacao foi encontrada
    """
    tavily_search = TavilySearchResults(max_results=3)
    search_docs = tavily_search.invoke(query)
    return search_docs

# 4- criacao do agente ReAct
tools = [search_web]
graph = create_react_agent(
    model,
    tools=tools,
    prompt=system_message
)

#export_graph = graph
