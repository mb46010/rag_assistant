import logging

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from redo.graph.node_answer import factory_answer_node
from redo.graph.node_intent import factory_intent_node
from redo.graph.node_query_hr import create_hr_query_node
from redo.graph.node_query_rag import factory_rag_query_node
from redo.graph.state import GraphState

logger = logging.getLogger(__name__)


def make_graph(llm):
    retry_policy = RetryPolicy(max_attempts=3)

    graph_builder = StateGraph(GraphState)
    graph_builder.add_node("intent", factory_intent_node(llm), retry_policy=retry_policy)
    graph_builder.add_node("query_hr", create_hr_query_node, retry_policy=retry_policy)
    graph_builder.add_node("query_rag", factory_rag_query_node(llm), retry_policy=retry_policy)
    graph_builder.add_node("answer", factory_answer_node(llm), retry_policy=retry_policy)

    graph_builder.add_edge(START, "intent")
    graph_builder.add_conditional_edges("intent", on_intent_edge)
    graph_builder.add_edge("query_hr", "answer")
    graph_builder.add_edge("query_rag", "answer")

    graph_builder.add_edge("answer", END)
    graph = graph_builder.compile()
    return graph


def on_intent_edge(state: GraphState) -> list[str]:
    intent = state.get("intent")
    logger.info("Intent Edge: %s", intent)
    responses = {
        "hr_system_query": ["query_hr"],
        "rag_query": ["query_rag"],
        "hr_rag_query": ["query_hr", "query_rag"],
        "out_of_scope": ["answer"],
    }
    parallel_nodes = responses.get(intent)
    if not parallel_nodes:
        raise ValueError(f"Invalid or missing intent: {intent}")
    return parallel_nodes
