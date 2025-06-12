from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from dotenv import load_dotenv
import os

from langgraph.graph import StateGraph
from langchain_core.runnables.graph import MermaidDrawMethod
from pydantic import BaseModel

# 1 - Carrega a API Key
load_dotenv()
API_KEY = os.getenv("API_KEY")

# 2 - definir o modelo
llm_model = ChatOpenAI(model="gpt-3.5-turbo", api_key=API_KEY)

# 3 - Define o prompt do sistema - A agente precisa se munir de várias tool para ficar power
system_message = SystemMessage(content = """
Você é um assistente. Se o usuário pedir contas, use a ferramenta 'somar'. Caso contrário, apenas responda normalmente
"""
)

# definindo a ferramenta de soma
@tool("somar")
def somar(valores: str) -> str:
    """soma dois numeros separados por virgula"""
    try:
        a, b = map(float, valores.split(","))
        return str(a + b)
    except Exception as e:
        return f"Erro ao somar: {str(e)}"

# 5 - Criação do Agente com LangGraph
tools = [somar]
graph = create_react_agent(
    model=llm_model,
    tools=tools,
    prompt=system_message
)

export_graph = graph

# 6 - extrair a resposta final
def extrair_resposta_final(result):
    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage) and m.content]
    if ai_messages:
        return ai_messages[-1].content
    else:
        return "Nenhuma resposta final encontrada"

# 7 - testando o agente
if __name__ == "__main__":
    entrada1 = HumanMessage(content="Quanto é 8 + 5")
    result1 = export_graph.invoke({"messages":[entrada1]})
    resposta_texto_1 = extrair_resposta_final(result1)
    print("Resposta 1: ", resposta_texto_1)
    print()

    entrada2 = HumanMessage(content="Quem pintou a monalise?")
    result2 = export_graph.invoke({"messages":[entrada2]})
    resposta_texto_2 = extrair_resposta_final(result2)
    for m in result2["messages"]:
        print(m)

    print("Resposta 2: ", resposta_texto_2)
    print()
