pip install langgraph langchain-openai pydantic


To implement a production-grade LangGraph workflow with checkpoint recovery, you need a state graph coupled with a persistent storage checkpointer. Checkpoints save the state of your agent after every node execution. If a run fails due to network drops, API timeouts, or crashes, you can resume execution from the exact point of failure without re-running previous expensive nodes.
As of recent LangGraph versions, the core persistent checkpointer is MemorySaver (for in-memory state tracking) or SqliteSaver / PostgresSaver (for production persistence) from the langgraph.checkpoint packages.

Project Dependencies

Install the required packages:
bash
pip install langgraph langchain-openai pydantic
Use code with caution.

Core LangGraph Workflow Implementation

This implementation sets up an agentic flow: Input Processor → LLM Generator → Evaluator (which conditional-routes back if validation fails). We use SqliteSaver to ensure state survives app restarts.
python
import os
import sqlite3
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI

# 1. Define the Global State Schema
class AgentState(TypedDict):
    input_query: str
    generated_response: str
    critique: str
    iterations: int
    status: str

# 2. Define the Node Implementations
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

async def process_input_node(state: AgentState) -> dict:
    print("\n--- [Executing Node: Input Processor] ---")
    # Initialize iteration counter if not present
    current_iters = state.get("iterations", 0)
    return {
        "iterations": current_iters + 1, 
        "status": "processing"
    }

async def generate_response_node(state: AgentState) -> dict:
    print("--- [Executing Node: LLM Generator] ---")
    prompt = f"Answer this query professionally: {state['input_query']}. Current context state: {state.get('critique', 'None')}"
    
    # Simulate potential infrastructure crashes or timeouts to test checkpoint recovery
    if state["iterations"] == 1 and os.environ.get("SIMULATE_CRASH") == "true":
        print("💥 CRASH SIMULATION: Simulating unexpected infrastructure failure...")
        raise SystemError("Unexpected connection drop to backend LLM host clusters.")
        
    response = await llm.ainvoke(prompt)
    return {"generated_response": str(response.content)}

async def evaluate_output_node(state: AgentState) -> dict:
    print("--- [Executing Node: Evaluator] ---")
    response_to_check = state["generated_response"]
    
    # Quick, programmatic validation (e.g., length check or guardrail check)
    if len(response_to_check) < 20:
        return {
            "critique": "Response too brief. Expand with comprehensive structural context.", 
            "status": "rejected"
        }
    return {"status": "approved"}

# 3. Conditional Router Logic
def route_post_evaluation(state: AgentState) -> Literal["generate", "__end__"]:
    if state["status"] == "rejected" and state["iterations"] < 3:
        print("🔄 Route Decision: Rejected. Routing back to Generator Node.")
        return "generate"
    print("🏁 Route Decision: Approved or Max iterations hit. Routing to END.")
    return "__end__"

# 4. Build and Compile the Graph with Persistence Storage (Checkpointer)
# Connect to a local SQLite instance for robust process-level persistence
conn = sqlite3.connect("agent_checkpoints.db", check_same_thread=False)
memory_checkpointer = SqliteSaver(conn)

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("process_input", process_input_node)
workflow.add_node("generate", generate_response_node)
workflow.add_node("evaluate", evaluate_output_node)

# Set up Edges
workflow.add_edge(START, "process_input")
workflow.add_edge("process_input", "generate")
workflow.add_edge("generate", "evaluate")

# Add Conditional Routing from the Evaluator
workflow.add_conditional_edges(
    "evaluate",
    route_post_evaluation,
    {
        "generate": "generate",
        "__end__": END
    }
)

# Compile graph with active checkpointer
app_graph = workflow.compile(checkpointer=memory_checkpointer)
Use code with caution.

Verification and Checkpoint Recovery Walkthrough

The script below demonstrates running the workflow, simulating a mid-flight crash, and completely restoring operations using the unique thread_id session identifier.
python
import asyncio

async def run_simulation():
    # Configure the persistent identifier config mapping
    thread_config = {"configurable": {"thread_id": "session_user_9921"}}
    
    # --- PHASE 1: Trigger Graph Execution with Crash Activated ---
    os.environ["SIMULATE_CRASH"] = "true"
    initial_state = {"input_query": "Explain quantum computing in one short sentence."}
    
    print("🚀 Starting original run thread...")
    try:
        async for event in app_graph.astream(initial_state, config=thread_config):
            print(f"Stream Step output event: {event}")
    except SystemError as err:
        print(f"\nCaught Simulated Runtime Error: {err}")
        print("✅ State safely saved at the last successful node execution snapshot point.")

    # --- PHASE 2: Resume Graph from Checkpoint Storage ---
    print("\n--- Restoring Infrastructure State ---")
    os.environ["SIMULATE_CRASH"] = "false" # Fix the issue/clear crash toggle
    
    print(f"🔮 Resuming execution for thread_id: {thread_config['configurable']['thread_id']}...")
    
    # We pass None as the input parameters because the graph state is recovered from SQLite using the thread_id
    async for event in app_graph.astream(None, config=thread_config):
        print(f"Resume Step output event: {event}")

    # --- PHASE 3: Inspect Final Restored State ---
    final_state = await app_graph.aget_state(thread_config)
    print("\n🏆 Final Agent State Retrieved from DB:")
    print(f"Iterations: {final_state.values.get('iterations')}")
    print(f"Status: {final_state.values.get('status')}")
    print(f"Response: {final_state.values.get('generated_response')}")

if __name__ == "__main__":
    # Ensure a clean DB environment for testing
    if os.path.exists("agent_checkpoints.db"):
        os.remove("agent_checkpoints.db")
        
    asyncio.run(run_simulation())
Use code with caution.

Key Recovery Mechanics

• thread_id Architecture: Every uniquely initialized execution sequence relies on a thread_id. Passing this configuration block allows LangGraph to match history trees inside the SQLite file.
• Resuming execution via astream(None, config): When you invoke astream or ainvoke with None as the first argument, LangGraph looks up the latest checkpoint associated with that thread_id and resumes execution from the exact node that failed or was paused.
To tailer this further to your system requirements, let me know:
• Do you want to implement Human-in-the-loop (breakpoints) where execution automatically pauses before the evaluation node to await manual developer/user approval?
• Are you planning to migrate this state management architecture to a distributed cluster setup using PostgresSaver or Redis?
