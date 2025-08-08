import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import Tool
from dotenv import load_dotenv
import os
#from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp import ClientSession, StdioServerParameters, stdio_client

# 1- Carrega API Key
load_dotenv()
API_KEY = os.getenv("API_KEY")

# 2- Definição do modelo
#model = ChatOpenAI(model="o4-mini-2025-04-16", api_key=API_KEY)
model = ChatOpenAI(model="gpt-4o-mini", api_key=API_KEY)

# 3 define o prompt do sistema
system_message = SystemMessage(content="""
 você é um assistente especializado em fornecer informações
 Sobre comunidades de Python para GenAI
 
 Ferramentas disponíveis no MCP Server:
 
 1. get_communit(location: str) ->
 - Função: retorna a melhor comunidade de Python para GenAI.
- Parâmetro: location (string)
- Retorno: "Code TI"

Seu papel é ser um intermediário direto entre o usuários e 
a ferramenta MCP, retornando apenas o resultado final das ferramentas.
 """)

# async def agent_mcp():
#     async with MultiServerMCPClient({
#         "code":{
#             "command": "python",
#             "args": ["./mcp_server.py"],
#             "transport": "stdio"
#         }
#     }) as client:
#         tools = await client.get_tools()
#         agent = create_react_agent(model, tools=tools, prompt=system_message, checkpointer=MemorySaver())
#         await agent.invoke({})


def convert_mcp_tools_to_langchain(mcp_tools, session):
    """Converte ferramentas MCP para formato LangChain - versão simplificada"""
    langchain_tools = []
    
    for mcp_tool in mcp_tools.tools:
        def make_tool_func(tool_name):
            """Cria função para ferramenta específica"""
            def tool_func(input_data: str = "") -> str:
                try:
                    # Executa de forma síncrona
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    # Chama ferramenta MCP
                    result = loop.run_until_complete(
                        session.call_tool(tool_name, {"location": input_data})
                    )
                    
                    # Extrai texto
                    text_results = []
                    for content in result.content:
                        if hasattr(content, 'text'):
                            text_results.append(content.text)
                        else:
                            text_results.append(str(content))
                    
                    loop.close()
                    return "\n".join(text_results)
                    
                except Exception as e:
                    return f"Erro ao executar {tool_name}: {str(e)}"
            
            return tool_func
        
        # Cria ferramenta LangChain
        langchain_tool = Tool(
            name=mcp_tool.name,
            description=mcp_tool.description or f"Ferramenta MCP: {mcp_tool.name}",
            func=make_tool_func(mcp_tool.name)
        )
        
        langchain_tools.append(langchain_tool)
    
    return langchain_tools

async def main():
    params=StdioServerParameters(
        command="python",        # The command to run your server
        args=['mcp_server.py'],  # Arguments to the command
    )

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            
            # Debug: mostra as ferramentas
            print("Ferramentas encontradas:")
            for tool in tools.tools:
                print(f"- {tool.name}: {tool.description}")
            #agent = create_react_agent(model, tools=tools, prompt=system_message, checkpointer=MemorySaver())
            #await agent.invoke({})
            # Testa chamada direta da ferramenta
            # result = await session.call_tool("get_community", {"location": "São Paulo"})
            # print("Resultado direto:", result)

            # Converte para formato LangChain
            langchain_tools = convert_mcp_tools_to_langchain(tools, session)
            
            # Cria agent
            agent = create_react_agent(
                model, 
                tools=langchain_tools, 
                prompt=system_message, 
                checkpointer=MemorySaver()
            )
            
            # Testa o agent
            response = await agent.ainvoke({
                "messages": ["Qual é a melhor comunidade Python para GenAI em São Paulo?"]
            })
            
            print("Resposta do agent:", response)
            
if __name__ == "__main__":
    asyncio.run(main())