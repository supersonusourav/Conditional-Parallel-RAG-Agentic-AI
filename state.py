from typing import TypedDict, Annotated, List, Dict
from langgraph.graph.message import add_messages

def combine_contexts(existing: Dict[str, str], new: Dict[str, str]) -> Dict[str, str]:
    """Reducer function to merge retrieved contexts from parallel branches."""
    res = dict(existing) if existing else {}
    res.update(new)
    return res

class State(TypedDict):
    programme: str
    messages: Annotated[list, add_messages]
    categories: List[str]  # e.g., ["academic", "fee"] or ["general"]
    retrieved_contexts: Annotated[Dict[str, str], combine_contexts]