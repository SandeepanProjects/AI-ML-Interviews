To implement a production-grade Human-in-the-Loop (HITL) workflow in LangGraph, you use the interrupt_before or interrupt_after configuration properties during graph compilation.
When an interrupt is triggered, LangGraph securely pauses execution, saves the complete state context to your checkpointer database, and yields control back to the application thread. The workflow remains paused until a human reviews the state, updates it if necessary, and explicitly instructs the graph to resume using its unique thread_id.

Project Dependencies

Ensure you have the required packages installed:
bash
pip install langgraph langchain-openai pydantic
Use code with caution.

Core LangGraph HITL Implementation

Below is a complete implementation of a copy-editing workflow. The agent processes an initial input draft, automatically pauses for human editorial approval, and gracefully resumes once human feedback or approval is submitted.
python
import os
import sqlite3
from typing import TypedDict, Literal
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI

# 1. Define the Shared State Schema
class WorkflowState(TypedDict):
    draft: str
    feedback: str
    approved: bool
    review_notes: str

# 2. Initialize the LLM Node Functions
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

async def draft_content_node(state: WorkflowState) -> dict:
    print("\n🤖 [Node: Draft Generator] Processing/Refining content...")
    
    prompt = f"Write a professional email announcement based on this draft: '{state['draft']}'."
    if state.get("feedback"):
        prompt += f" Incorporate this user feedback: '{state['feedback']}'."
        
    response = await llm.ainvoke(prompt)
    return {"draft": str(response.content), "approved": False}

async def human_review_placeholder_node(state: WorkflowState) -> dict:
    """
    This node serves as the formal state bridge for Human Interaction. 
    The graph will be interrupted BEFORE hitting this node.
    """
    print("👀 [Node: Human Review Bridge] Executing verification passes...")
    return {}

# 3. Conditional Router Logic
def route_post_review(state: WorkflowState) -> Literal["draft_content", "__end__"]:
    if state.get("approved") is True:
        print("✅ Router Decision: Content Approved. Finalizing execution thread.")
        return "__end__"
    print("🔄 Router Decision: Content Rejected/Modified. Routing back to Draft Generator.")
    return "draft_content"

# 4. Assemble the Persistent Graph Configuration
conn = sqlite3.connect("hitl_workflow.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

workflow = StateGraph(WorkflowState)

# Define nodes
workflow.add_node("draft_content", draft_content_node)
workflow.add_node("human_review", human_review_placeholder_node)

# Connect edges
workflow.add_edge(START, "draft_content")
workflow.add_edge("draft_content", "human_review")

# Apply conditional routing exit pathways from the review node
workflow.add_conditional_edges(
    "human_review",
    route_post_review,
    {
        "draft_content": "draft_content",
        "__end__": END
    }
)

# CRITICAL STEP: Compile with an interrupt checkpoint hook directly BEFORE the human review node
app_graph = workflow.compile(
    checkpointer=checkpointer,
    interrupt_before=["human_review"]
)
Use code with caution.

Verification and Workflow Resumption Walkthrough

The following script simulates a multi-step user engagement session: running the graph until it automatically halts, inspecting the state, applying an out-of-band human approval correction, and resuming the execution thread cleanly.
python
import asyncio

async def run_hitl_simulation():
    # Context trace configuration mapping
    thread_config = {"configurable": {"thread_id": "marketing_campaign_101"}}
    
    # --- PHASE 1: Initial Automated Execution ---
    initial_input = {"draft": "Hey team, launch is delayed to Friday because of some bugs."}
    print("🚀 System: Initiating workflow execution stream...")
    
    async for event in app_graph.astream(initial_input, config=thread_config):
        print(f"Stream Progress: {event}")
        
    # --- PHASE 2: Verify the Interrupt State ---
    state_snapshot = await app_graph.aget_state(thread_config)
    print(f"\n⏸️ Graph State Paused. Next scheduled node to execute: {state_snapshot.next}")
    print(f"Current Draft Content for Review:\n---\n{state_snapshot.values.get('draft')}\n---")
    
    # --- PHASE 3: Human Interaction (Reject & Provide Feedback) ---
    print("\n👨‍💻 Human Action: Reviewer rejects draft, requesting structural edits...")
    
    # Update the snapshot state explicitly using the thread config mapping
    await app_graph.aupdate_state(
        thread_config,
        {"approved": False, "feedback": "Make it sound more optimistic and enthusiastic!"},
        as_node="human_review" # Specify the node namespace applying this update
    )
    
    # Resume the graph from the break point (passing None tells it to read from the checkpoint)
    print("▶️ System: Resuming workflow with new feedback payload...")
    async for event in app_graph.astream(None, config=thread_config):
        print(f"Stream Progress: {event}")

    # --- PHASE 4: Second Interrupt (Reviewing the Refined Draft) ---
    state_snapshot = await app_graph.aget_state(thread_config)
    print(f"\n⏸️ Graph State Paused Again. Next scheduled node: {state_snapshot.next}")
    print(f"Refined Draft Content for Review:\n---\n{state_snapshot.values.get('draft')}\n---")

    # --- PHASE 5: Human Interaction (Approve) ---
    print("\n👨‍💻 Human Action: Reviewer approves the new content draft.")
    await app_graph.aupdate_state(
        thread_config,
        {"approved": True, "feedback": ""},
        as_node="human_review"
    )
    
    print("▶️ System: Resuming workflow for finalization pass...")
    async for event in app_graph.astream(None, config=thread_config):
        print(f"Stream Progress: {event}")
        
    print("\n🏁 Workflow execution successfully reached its termination state.")

if __name__ == "__main__":
    # Reset localized checkpoint DB files for demo reproducibility
    if os.path.exists("hitl_workflow.db"):
        os.remove("hitl_workflow.db")
        
    asyncio.run(run_hitl_simulation())
Use code with caution.

Core Architecture Mechanics

• interrupt_before=["human_review"]: Instructs the LangGraph execution engine to pause processing immediately prior to executing the specified node. The state engine locks the state snapshot inside the SQLite file and releases the CPU process thread.
• aupdate_state: Simulates the human submission action. It modifies the current values inside the database for that specific thread_id session, allowing routers (route_post_review) to make new routing evaluations upon resumption.
• Resuming execution via astream(None, config): Passing None as the input payload instructs LangGraph to retrieve the active paused state snapshot directly from the checkpointer database and resume execution where it left off.
To ready this architecture for your application, let me know:
• Do you plan to build a web frontend (FastAPI / REST API wrapper) where human operators can inspect and submit approvals via endpoint calls?
• Do you need to track a full audit history log of every change made by different human reviewers across multiple revisions?
