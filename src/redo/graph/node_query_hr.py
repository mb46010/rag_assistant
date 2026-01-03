import logging
from typing import Any, Dict

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from redo.graph.state import GraphState


class HRSystemResponse(TypedDict):
    holiday_balance: int


logger = logging.getLogger(__name__)


async def mock_return_holidays(user_email: str) -> Dict[str, Any]:
    return {"holiday_balance": 10}


async def create_hr_query_node(state: GraphState) -> HRSystemResponse:
    user_email = state["user_email"]
    hr_system_response = await mock_return_holidays(user_email)
    return {"hr_system_response": hr_system_response}
