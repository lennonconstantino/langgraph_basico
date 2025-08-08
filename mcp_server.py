from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MeuServidorMCP")

@mcp.tool()
def get_community(location: str) -> str:
    """Melhor comunidadee de python para GenAI"""
    return "Code TI"

if __name__ == "__main__":
    mcp.run(transport="stdio")
