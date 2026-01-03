"""Module for the intent classification node."""

import logging
from typing import Any, Dict, List, Literal

from langchain_core.messages import SystemMessage
from langfuse import observe
from pydantic import BaseModel, Field

from redo.graph.state import GraphState
from redo.prompts.intent import INTENT_PROMPT

logger = logging.getLogger(__name__)


class IntentResponse(BaseModel):
    """The structured response for intent classification."""

    intent: Literal["hr_system_query", "rag_query", "hr_rag_query", "out_of_scope"] = Field(
        ..., description="The intent of the user"
    )


def factory_intent_node(llm):
    """Create an intent node."""

    @observe()
    def create_intent_node(state: GraphState) -> Dict[str, Any]:
        """Classify the user intent using the LLM."""
        logger.info("Intent Node started")
        system_message = SystemMessage(content=INTENT_PROMPT)
        messages = [system_message] + state.get("messages", [])

        model = llm.with_structured_output(IntentResponse)
        try:
            response = model.invoke(messages)
        except Exception as e:
            logger.exception("LLM Error: %s", str(e))
            return {"error": {"type": "llm_error", "message": str(e)}}

        try:
            intent = IntentResponse.model_validate(response.model_dump()).intent
        except Exception as e:
            logger.error("Intent validation failed: %s", str(e))
            return {"error": {"type": "validation_error", "message": str(e)}}

        logger.info("Intent Node finished, with intent: %s", intent)
        return {"intent": intent}

    return create_intent_node
