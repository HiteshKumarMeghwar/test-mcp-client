import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import ToolMessage
from dotenv import load_dotenv
import json
load_dotenv()

model_gen = HuggingFaceEndpoint(
    # repo_id="Qwen/Qwen2.5-7B-Instruct",
    # repo_id="google/gemma-2-2b-it",
    repo_id="openai/gpt-oss-20b",
    # repo_id="MiniMaxAI/MiniMax-M2",
    # repo_id="meta-llama/Llama-3.1-70B-Instruct",
    # repo_id="moonshotai/Kimi-K2-Thinking",
    task="text-generation"
)
generator_llm = ChatHuggingFace(llm=model_gen)

SERVERS = {
    "expense": {
        "transport": "stdio",
        "command": "uv",
        "args": [
            "run",
            "--with",
            "fastmcp",
            "fastmcp",
            "run",
            "C:\\Users\\DEVELOPER-\\Desktop\\expense-tracker-mcp\\main.py"
        ]
    },
    "math": {
        "transport": "streamable_http",  # if this fails, try 'sse'
        "url": "https://test-remote-server2323.fastmcp.app/mcp"
    }
}



async def main():
    client = MultiServerMCPClient(SERVERS)
    tools = await client.get_tools()

    named_tools = {}
    for tool in tools:
        named_tools[tool.name] = tool

    # make the llm tool-aware
    llm_with_tools = generator_llm.bind_tools(tools)

    prompt = "give a random number range will be bitween 22 and 70"
    response = await llm_with_tools.ainvoke(prompt)

    if not getattr(response, "tool_calls", None):
        print(response.content)
        return
    
    tool_messages = []
    for tc in response.tool_calls:

        selected_tool = tc["name"]
        selected_tool_args = tc.get("args") or {}
        selected_tool_id = tc["id"]

        tool_result = await named_tools[selected_tool].ainvoke(selected_tool_args)
        tool_messages.append(ToolMessage(tool_call_id=selected_tool_id, content=json.dumps(tool_result)))

    final_response = await llm_with_tools.ainvoke([prompt, response, *tool_messages])
    print("LLM Tools Response", final_response.content)
        

if __name__ == '__main__':
    asyncio.run(main())