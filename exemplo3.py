from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from pydantic import BaseModel
from langgraph.graph import StateGraph
import os

# 1 - Carrega a API Key
load_dotenv()
API_KEY = os.getenv("API_KEY")

# 2 - definir o modelo
llm_model = ChatOpenAI(model="gpt-3.5-turbo", api_key=API_KEY)

# 3 -defina o estado do graph
class GraphState(BaseModel):
    input: str
    output: str
    tipo: str

# 4 - funcao de realizar calculo
def realizar_calculo(state: GraphState )-> GraphState:
    return GraphState(input=state.input, output="Resposta de cálculo fictício: 42", tipo="")

# 5 - Funcao para responder perguntas normais
def responder_curiosidade(state: GraphState) -> GraphState:
    response = llm_model.invoke([HumanMessage(content=state.input)])
    return GraphState(input=state.input, output=response.content, tipo="")

# 6 - funcao para tratar perguntas nao reconhecidas
def responder_erro(state: GraphState) -> GraphState:
    return GraphState(input=state. input, output="Desculpe, não entendi sua pergunta.", tipo="")

# 7 - Função de classificação dos nodes
def classificar(state: GraphState) -> GraphState:
    pergunta = state.input.lower()
    if any(palavra in pergunta for palavra in ["soma", "quanto é", "+", "calcular"]):
        tipo = "calculo"
    elif any(palavra in pergunta for palavra in ["quem", "onde", "quando", "por que", "qual"]):
        tipo = "curiosidade"
    else:
        tipo = "desconhecido"
    return GraphState(input=state.input, output="", tipo=tipo)

# 8 - criando o graph e adicionando os nodes
graph = StateGraph(GraphState)
graph.add_node("classificar", classificar)
graph.add_node("realizar_calculo", realizar_calculo)
graph.add_node("responder_curiosidade", responder_curiosidade)
graph.add_node("responder_erro", responder_erro)

# 9 - Adicionando condicionais
graph.add_conditional_edges(
    "classificar",
    lambda state: {
        "calculo": "realizar_calculo",
        "curiosidade": "responder_curiosidade",
        "desconhecido" : "responder_erro",
    }[state.tipo]
)

# 10 - definindo entrada e saida e compilacao
graph.set_entry_point("classificar")
graph.set_finish_point(["realizar_calculo", "responder_curiosidade", "responder_erro"])
export_graph = graph.compile()

# 11 testando o projeto
if __name__ == "__main__":
    exemplos = [
        "Quanto é 10 + 5?",
        "Quem inventou a lampada?",
        "Me diga um comando especial"
    ]
    for exemplo in exemplos:
        result = export_graph.invoke(GraphState(input=exemplo, output="", tipo=""))
        output=result["output"]
        print(f"Pergunta: {exemplo}\nResposta: {output}\n")

# criar uma funcao e trabalhar como uma tool