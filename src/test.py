from graph import CRAGState, crag_app
from telemetry import logger

def run_test(question: str):
    print("\n" + "="*80)
    print(f"TEST QUERY: {question}")
    print("="*80 + "\n")
    
    # Initialize the input state
    initial_state: CRAGState = {
        "question": question,
        "original_question": question,
        "documents": [],
        "web_search_needed": False,
        "generation": "",
        "iteration_count": 0,
        "hallucination_verdict": None,
    }
    
    # Run the state machine
    final_state = crag_app.invoke(initial_state)
    
    print("\n" + "-"*40 + " FINAL RESULT " + "-"*40)
    print(f"Final Answer:\n{final_state['generation']}\n")
    print(f"Sources Used ({len(final_state['documents'])} chunks):")
    for doc in final_state["documents"]:
        src = doc.metadata.get("source", "unknown")
        cid = doc.metadata.get("chunk_id", "N/A")
        print(f" - [{cid}] from: {src}")
    print("-" * 94 + "\n")

if __name__ == "__main__":
    # Test Case 1: Internal Policy (Should pass grader, skip web search, answer with internal codes)
    run_test("What is the reimbursement cap and code for home office desks?")
    
    # Test Case 2: Out of Domain / Not in Manual (Should fail grader, route to Tavily web search)
    run_test("What is the current standard IRS business mileage rate for 2026?")