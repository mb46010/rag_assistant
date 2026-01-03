import logging
from typing import Any, Dict, List, Literal

from langchain_core.messages import SystemMessage
from langfuse import observe
from pydantic import BaseModel, Field

from redo.graph.state import GraphState
from redo.prompts.answer import ANSWER_PROMPT

logger = logging.getLogger(__name__)


class AnswerResponse(BaseModel):
    answer: str = Field(..., description="The answer to the user's message")


def fill_prompt(prompt: str, state: GraphState) -> str:
    import json

    intent = state.get("intent")
    if intent == "out_of_scope":
        context = "Intent: {intent}"
        return prompt.format(context=context)

    rag_result = state.get("rag_response", [])
    hr_result = state.get("hr_system_response", {})
    context = f"RAG Documents: {json.dumps(rag_result)}\nHR Data: {json.dumps(hr_result)}"
    return prompt.format(context=context)


def factory_answer_node(llm):
    """Create an answer node"""

    @observe()
    def create_answer_node(state: GraphState) -> Dict[str, Any]:
        logger.info("Answer Node started")

        filled_prompt = fill_prompt(ANSWER_PROMPT, state)

        system_message = SystemMessage(content=filled_prompt)
        messages = [system_message] + state.get("messages", [])

        model = llm.with_structured_output(AnswerResponse)
        try:
            response = model.invoke(messages)
        except Exception as e:
            logger.exception("LLM Error: %s", str(e))
            return {"error": {"type": "llm_error", "message": str(e)}}

        try:
            answer = AnswerResponse.model_validate(response.model_dump()).answer
        except Exception as e:
            logger.error("Answer validation failed: %s", str(e))
            return {"error": {"type": "validation_error", "message": str(e)}}

        logger.info("Answer Node finished, with answer: %s", answer)
        return {"final_answer": answer}

    return create_answer_node
