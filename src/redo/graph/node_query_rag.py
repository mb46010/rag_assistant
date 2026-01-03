import logging
from typing import Any, Dict, List

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from redo.graph.state import GraphState

hr_policies = [
    {
        "title": "Remote Work Policy",
        "content": "Employees may work remotely up to 3 days per week with manager approval. Full-time remote work requires VP approval. Equipment stipend: CHF 500/year.",
    },
    {
        "title": "Vacation Policy",
        "content": "Employees receive 25 days annual vacation. Vacation must be requested 2 weeks in advance. Unused vacation expires December 31st.",
    },
    {
        "title": "Parental Leave",
        "content": "Primary caregivers: 16 weeks paid leave. Secondary caregivers: 4 weeks paid leave. Must notify HR 3 months before expected date.",
    },
]


class RagResponse(TypedDict):
    retrieved_documents: List[Dict[str, Any]]


logger = logging.getLogger(__name__)


async def mock_return_documents(query: str) -> Dict[str, Any]:
    return hr_policies


async def create_rag_query_node(state: GraphState) -> RagResponse:
    rag_response = await mock_return_documents(state["messages"][-1].content)
    return {"rag_response": rag_response}
