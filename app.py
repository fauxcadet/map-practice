import asyncio
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from mcp_use import MCPAgent, MCPClient


async def run_memory_chat():
    """Run a chat using the MCP Agent's built-in conversation memory"""

    load_dotenv()

    # Load API key
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

    config_file = "browser_mcp.json"

    print("Initializing MCP Agent with built-in memory...")

    # ✅ fixed typo
    client = MCPClient.from_config_file(config_file)

    llm = ChatGroq(model="qwen-qwq-32b")

    # ✅ fixed bracket + formatting
    agent = MCPAgent(
        client=client,
        llm=llm,
        max_steps=15,
        memory_enabled=True,
    )

    print("\n=== Interactive MCP Chat ===")
    print("Type 'exit' to quit | 'clear' to reset memory")
    print("================================\n")

    try:
        while True:
            user_input = input("\nYou: ")

            if user_input.lower() == "exit":
                print("\n=== Chat ended ===")
                break

            if user_input.lower() == "clear":
                agent.clear_conversation_history()
                print("\n=== Conversation history cleared ===")
                continue

            print("\nAssistant:", end=" ", flush=True)

            try:
                response = await agent.run(user_input)
                print(response)

            except Exception as e:
                print(f"\nError: {e}")

    finally:
        # ✅ removed undefined `clean`
        if client.sessions:
            await client.close_all_sessions()


if __name__ == "__main__":
    asyncio.run(run_memory_chat())
