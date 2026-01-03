import argparse
import asyncio
import json
import os
from datetime import datetime

from dotenv import load_dotenv

from redo.graph.graph import make_graph
from redo.model import get_default_model

load_dotenv()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--user_message",
        type=str,
        default="how many holidays do I have?",
        required=False,
        help="The message to process",
    )
    parser.add_argument(
        "--user_email", type=str, default="user@example.com", required=False, help="The message to process"
    )
    args = parser.parse_args()

    state_config = {
        "messages": [{"content": args.user_message, "role": "user"}],
        "user_email": args.user_email,
    }

    llm = get_default_model()
    graph = make_graph(llm)

    result = asyncio.run(graph.ainvoke(state_config))
    print(result)

    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Serialize results to JSON with timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    # Use a custom encoder or handle non-serializable objects (like message objects)
    # LangGraph results often contain langchain message objects which need serialization
    def serialize(obj):
        if hasattr(obj, "dict"):
            return obj.dict()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)

    with open(filepath, "w") as f:
        json.dump(result, f, indent=4, default=serialize)
