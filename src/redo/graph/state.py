from typing import Any, Dict, Literal, Optional

from langchain.graph import StateGraph

# from langchain.messages import Message
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict


class GraphState(TypedDict):
    # input
    messages: Annotated[list, add_messages]
    user_email: str

    # progress
    intent: Literal["hr_system_query", "rag_query", "hr_rag_query"]
    user_information: Optional[Dict[str, Any]]
    hr_system_response: Optional[Dict[str, Any]]
    rag_response: Optional[Dict[str, Any]]

    # output
    final_answer: str

    # error
    error: Optional[Dict[str, Any]]
