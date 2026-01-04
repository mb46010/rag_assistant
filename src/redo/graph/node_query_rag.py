import logging
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import observe
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from redo.graph.state import GraphState
from redo.prompts.rag import CONDENSE_QUESTION_PROMPT
from redo.rag.retrieve import retrieve_policies

logger = logging.getLogger(__name__)


class RagResponse(TypedDict):
    retrieved_documents: List[Dict[str, Any]]


class StandaloneQuestion(BaseModel):
    """The standalone question for the RAG system."""

    question: str = Field(..., description="The standalone question")


def factory_rag_query_node(llm):
    @observe()
    async def create_rag_query_node(state: GraphState) -> RagResponse:
        logger.info("RAG Query Node started")

        # 1. Condense the question
        last_message = state["messages"][-1]
        chat_history = state["messages"][:-1]

        filled_prompt = CONDENSE_QUESTION_PROMPT.format(chat_history=chat_history, question=last_message.content)

        model = llm.with_structured_output(StandaloneQuestion)
        try:
            response = model.invoke([SystemMessage(content=filled_prompt)])
            query = response.question
            logger.info(f"Generated standalone question: {query}")
        except Exception as e:
            logger.error(f"Error generating standalone question: {e}")
            # Fallback to original message if summarization fails
            query = last_message.content

        # 2. Retrieve documents
        nodes = retrieve_policies(query)

        # 3. Format results
        retrieved_documents = []
        for node in nodes:
            retrieved_documents.append({"content": node.get_content(), "score": node.score, "metadata": node.metadata})

        return {"rag_response": retrieved_documents}

    return create_rag_query_node
