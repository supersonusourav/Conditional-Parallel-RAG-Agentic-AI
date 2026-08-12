import mlflow
from state import State
from graph import app

def run_cli():
    print("Welcome to the College Assistant!\n")
    print("Which programme are you in?")
    print("1. BCA\n2. BBA\n3. B.Com (H)")

    choice = input("\nEnter 1, 2 or 3: ")
    programme_map = {"1": "BCA", "2": "BBA", "3": "B.Com (H)"}
    student_programme = programme_map.get(choice.strip(), "BCA")

    print(f"\nGreat! You're set as a {student_programme} student.\n")

    with mlflow.start_run(run_name="Interactive_Session"):
        while True:
            user_query = input("You: ")
            if user_query.lower() in ["exit", "quit"]:
                break

            # Explicit State dictionary matching the TypedDict schema
            initial_state: State = {
                "programme": student_programme,
                "messages": [("human", user_query)],
                "categories": [],
                "retrieved_contexts": {}
            }

            result = app.invoke(initial_state)

            print(f"\nAssistant: {result['messages'][-1].content}\n")

if __name__ == "__main__":
    run_cli()