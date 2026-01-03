import argparse
import asyncio
import os

from dotenv import load_dotenv

from redo.graph.graph import make_graph
from redo.model import get_default_model

load_dotenv()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "user_message", type=str, default="how many holidays do I have?", help="The message to process"
    )
    parser.add_argument("user_email", type=str, default="user@example.com", help="The message to process")
    args = parser.parse_args()

    state_config = {
        "messages": [{"content": args.user_message, "role": "user"}],
        "user_email": args.user_email,
    }

    llm = get_default_model()
    graph = make_graph(llm)

    result = asyncio.run(graph.ainvoke(state_config))
    print(result)
