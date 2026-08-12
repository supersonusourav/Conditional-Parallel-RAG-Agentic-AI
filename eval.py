import json
import pandas as pd
import mlflow

# MLflow 3 uses scorers for genai.evaluate
from mlflow.genai.scorers import RelevanceToQuery
from mlflow.metrics.genai import faithfulness, answer_relevance

from graph import app
from state import State

# Local Ollama judge model URI
JUDGE_MODEL = "ollama:/gemma4:e4b"

mlflow.set_experiment("College_Assistant_RAG")

# Prediction wrapper for your agent
def predict_fn(inputs: dict) -> dict:
    initial_state: State = {
        "programme": inputs.get("programme", ""),
        "messages": [("human", inputs["inputs"])],
        "categories": [],
        "retrieved_contexts": {}
    }

    result = app.invoke(initial_state)

    ai_message = result["messages"][-1].content
    combined_context = "\n".join(
        [ctx for ctx in result.get("retrieved_contexts", {}).values() if ctx != "NO_RETRIEVAL_NEEDED"]
    )

    return {
        "response": ai_message,
        "context": combined_context if combined_context else "No context retrieved."
    }

def run_evaluation():
    with open("eval_dataset.json", "r") as f:
        raw_data = json.load(f)
    
    eval_df = pd.DataFrame(raw_data)

    if "ground_truth_answer" in eval_df.columns:
        eval_df = eval_df.rename(columns={"ground_truth_answer": "expectations"})

    with mlflow.start_run(run_name="Local_Ollama_GenAI_Evaluation"):
        # Correct parameter in MLflow 3 is 'scorers', not 'metrics'
        eval_results = mlflow.genai.evaluate(
            data=eval_df,
            predict_fn=predict_fn,
            scorers=[
                faithfulness(model=JUDGE_MODEL),
                answer_relevance(model=JUDGE_MODEL)
            ]
        )

        print("\n--- Evaluation Successful ---")
        print(eval_results.metrics)

if __name__ == "__main__":
    run_evaluation()