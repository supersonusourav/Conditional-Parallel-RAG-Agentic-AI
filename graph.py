from langgraph.graph import StateGraph, START, END
from state import State
from nodes import (
    classifier_node,
    academic_rag_node,
    fee_rag_node,
    general_node,
    response_node
)

def route_query(state: State) -> list:
    """Returns a list of node names to execute in parallel based on classification."""
    categories = state.get("categories", ["general"])
    targets = []
    
    if "academic" in categories:
        targets.append("academic_rag")
    if "fee" in categories:
        targets.append("fee_rag")
    if "general" in categories or not targets:
        targets.append("general")
        
    return targets

# Build Graph
builder = StateGraph(State)

builder.add_node("classifier", classifier_node)
builder.add_node("academic_rag", academic_rag_node)
builder.add_node("fee_rag", fee_rag_node)
builder.add_node("general", general_node)
builder.add_node("response", response_node)

# Edges
builder.add_edge(START, "classifier")

# Conditional parallel dispatch
builder.add_conditional_edges(
    "classifier",
    route_query,
    ["academic_rag", "fee_rag", "general"]
)

# Fan-in convergence to response node
builder.add_edge("academic_rag", "response")
builder.add_edge("fee_rag", "response")
builder.add_edge("general", "response")

builder.add_edge("response", END)

app = builder.compile()