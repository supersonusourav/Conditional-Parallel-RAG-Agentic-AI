import json
from config import llm
from retriever import academic_retriever, fee_retriever
from state import State

def classifier_node(state: State) -> dict:
    """Classifies user intent into one or more categories for parallel routing."""
    last_message = state['messages'][-1].content

    prompt = (
        "Analyze the following query and classify it into one or more categories from: ['academic', 'fee', 'general'].\n\n"
        "- 'academic': Attendance, exams, grading, credits, course structure, subjects, or degree requirements.\n"
        "- 'fee': Tuition, payments, refunds, late charges, or scholarships.\n"
        "- 'general': Greetings or general conversation.\n\n"
        "Return ONLY a JSON list of strings. Example: [\"academic\", \"fee\"]\n\n"
        f"Query: {last_message}"
    )

    response = llm.invoke(prompt)
    try:
        raw_content = str(response.content).strip()
        # Find JSON boundaries if extra text is produced
        start = raw_content.find('[')
        end = raw_content.rfind(']') + 1
        categories = json.loads(raw_content[start:end]) if start != -1 and end != 0 else ["general"]
    except Exception:
        categories = ["general"]

    return {"categories": categories, "retrieved_contexts": {}}


def academic_rag_node(state: State) -> dict:
    """Retrieves context from the academic handbook."""
    query = state["messages"][-1].content
    docs = academic_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_contexts": {"academic": context}}


def fee_rag_node(state: State) -> dict:
    """Retrieves context from the fee structure document."""
    query = state["messages"][-1].content
    docs = fee_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_contexts": {"fee": context}}


def general_node(state: State) -> dict:
    """Handles general chit-chat queries."""
    return {"retrieved_contexts": {"general": "NO_RETRIEVAL_NEEDED"}}


def response_node(state: State) -> dict:
    """Synthesizes all retrieved contexts (academic, fee, general) into a coherent response."""
    query = state["messages"][-1].content
    programme = state.get("programme", "Unknown")
    contexts = state.get("retrieved_contexts", {})

    combined_context_str = ""
    for category, ctx in contexts.items():
        if ctx != "NO_RETRIEVAL_NEEDED":
            combined_context_str += f"\n--- Context ({category.upper()}) ---\n{ctx}\n"

    if not combined_context_str.strip():
        prompt = (
            f"You are a friendly college assistant talking to a {programme} student. "
            f"Answer this question using your knowledge:\n\n{query}"
        )
    else:
        prompt = (
            f"You are a college assistant helping a {programme} student. "
            f"Use the following contexts from official college documents to answer the question accurately. "
            f"Highlight specific details relevant to the {programme} programme.\n\n"
            f"{combined_context_str}\n"
            f"Question: {query}\n\n"
            f"Give a clear, comprehensive, and friendly answer addressing all parts of the question."
        )

    response = llm.invoke(prompt)
    return {"messages": [("ai", str(response.content).strip())]}