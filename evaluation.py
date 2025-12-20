import uuid
from typing import Dict, Any
from langsmith import evaluate, Client
from src.rag.rag_agent import Agent

def get_input_query(inputs: Dict[str, Any]) -> str:
    """Extract the user query from dataset inputs using common keys."""
    query = inputs.get("question") or inputs.get("input")
    
    # Fallback: search for the first string value if standard keys are missing
    if not query:
        query = next((v for v in inputs.values() if isinstance(v, str)), None)
    
    if not query:
        raise ValueError(f"Could not find query in inputs. Available keys: {list(inputs.keys())}")
    return query

def run_evaluation(dataset_name: str, experiment_prefix: str, max_concurrency: int):
    """Run the LangSmith evaluation suite."""
    print("--- Initializing Agent ---")
    rag_agent = Agent()
    client = Client()

    def target(inputs: Dict[str, Any]) -> Dict[str, str]:
        # 1. Prepare Input
        query = get_input_query(inputs)
        print(f"👉 Processing: {query[:50]}...")

        # 2. Session Isolation
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        
        # 3. Agent Execution
        response = rag_agent.agent.invoke(
            {"messages": [{"role": "user", "content": query}]}, 
            config=config
        )

        # 4. Extract Answer from Message Objects
        messages = response.get("messages", [])
        if not messages:
            return {"output": "Error: No response generated"}
            
        content = messages[-1].content
        
        # Handle both string and complex list content types
        if isinstance(content, list):
            answer = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        else:
            answer = str(content)

        return {"output": answer}

    print(f"🚀 Starting Eval: {dataset_name}")

    try:
        results = evaluate(
            target,
            data=dataset_name,
            evaluators=[],  # Built-in 'Correctness' evaluator triggered via LangSmith UI
            experiment_prefix=experiment_prefix,
            max_concurrency=max_concurrency,
            client=client
        )
        print(f"\n✅ Evaluation Task Submitted! Check results in LangSmith UI.")
        return results
        
    except Exception as e:
        print(f"\n❌ Evaluation Failed: {e}")
        raise

if __name__ == "__main__":
    
    DATASET_NAME = "ds-enchanted-thump-19"
    EXPERIMENT_PREFIX = "rag-agent-eval"
    MAX_CONCURRENCY = 1

    run_evaluation(dataset_name=DATASET_NAME, experiment_prefix=EXPERIMENT_PREFIX, max_concurrency=MAX_CONCURRENCY)