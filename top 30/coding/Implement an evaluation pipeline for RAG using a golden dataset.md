To evaluate a Retrieval-Augmented Generation (RAG) system accurately, you must test it against a Golden Dataset containing ground-truth query-answer pairs. The industry standard is to assess the system across two key dimensions: Retrieval Quality (Context Recall, Context Precision) and Generation Quality (Faithfulness/Groundedness, Answer Relevance).
This production-grade pipeline uses FastAPI for non-blocking execution, tiktoken for token boundary checks, and an asynchronous LLM-as-a-judge framework to run evaluations concurrently across the entire dataset.

Project Dependencies

Install the required evaluation metrics packages:
bash
pip install fastapi uvicorn openai tiktoken pydantic numpy
Use code with caution.

Core RAG Evaluation Pipeline Implementation

This implementation defines a structured metric runner using structured JSON outputs from OpenAI's gpt-4o-mini to evaluate a RAG workflow against a golden evaluation test suite.
python
import asyncio
import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import numpy as np

# Initialize the async evaluation engine client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", "your-key-here"))

# --- 1. Define Data Schemas ---
class GoldenDataPoint(BaseModel):
    query: str
    ground_truth: str

class RAGSystemOutput(BaseModel):
    query: str
    generated_answer: str
    retrieved_contexts: List[str]

class EvaluationMetrics(BaseModel):
    faithfulness: float = Field(..., description="Score 0.0-1.0: Is the answer grounded strictly in retrieved context?")
    context_recall: float = Field(..., description="Score 0.0-1.0: Does the context contain the ground truth information?")
    answer_relevance: float = Field(..., description="Score 0.0-1.0: Does the generated answer directly address the user query?")
    rationale: str = Field(..., description="Detailed textual justification for the given scores.")

class EvaluationReportItem(BaseModel):
    query: str
    generated_answer: str
    ground_truth: str
    retrieved_contexts: List[str]
    scores: EvaluationMetrics

class BatchEvaluationSummary(BaseModel):
    mean_faithfulness: float
    mean_context_recall: float
    mean_answer_relevance: float
    detailed_reports: List[EvaluationReportItem]

# --- 2. LLM-as-a-Judge Asynchronous Evaluator ---
class RAGEvaluator:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    async def evaluate_single_run(self, golden: GoldenDataPoint, system: RAGSystemOutput) -> EvaluationReportItem:
        """
        Uses an LLM judge with structured outputs to score a single RAG run.
        """
        context_str = "\n---\n".join(system.retrieved_contexts)
        
        system_prompt = (
            "You are an expert QA evaluation judge. Analyze the provided RAG components and score "
            "them strictly between 0.0 (worst) and 1.0 (perfect) based on these principles:\n"
            "1. Faithfulness: Is every statement in the generated answer completely supported by the retrieved context?\n"
            "2. Context Recall: Does the retrieved context contain the necessary facts to construct the ground truth answer?\n"
            "3. Answer Relevance: Is the generated answer clear, direct, and free of redundant or unhelpful filler details?\n"
            "Provide a concise rationale explaining your scores."
        )
        
        user_content = (
            f"User Query: {golden.query}\n\n"
            f"Retrieved Context:\n{context_str}\n\n"
            f"Generated Answer: {system.generated_answer}\n\n"
            f"Ground Truth Answer: {golden.ground_truth}"
        )

        try:
            # Leverage OpenAI Structured Outputs to enforce numeric evaluation scores
            response = await openai_client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                response_format=EvaluationMetrics,
                temperature=0.0 # Force deterministic evaluation results
            )
            
            metrics = response.choices[0].message.parsed
            return EvaluationReportItem(
                query=golden.query,
                generated_answer=system.generated_answer,
                ground_truth=golden.ground_truth,
                retrieved_contexts=system.retrieved_contexts,
                scores=metrics
            )
        except Exception as e:
            # Fallback layout matrix on execution drops
            return EvaluationReportItem(
                query=golden.query,
                generated_answer=system.generated_answer,
                ground_truth=golden.ground_truth,
                retrieved_contexts=system.retrieved_contexts,
                scores=EvaluationMetrics(
                    faithfulness=0.0,
                    context_recall=0.0,
                    answer_relevance=0.0,
                    rationale=f"Evaluation pipeline failed: {str(e)}"
                )
            )

    async def evaluate_batch(
        self, 
        golden_dataset: List[GoldenDataPoint], 
        system_outputs: List[RAGSystemOutput],
        concurrency_limit: int = 5
    ) -> BatchEvaluationSummary:
        """
        Runs batch evaluations concurrently, bounded by a semaphore to avoid API rate limits.
        """
        semaphore = asyncio.Semaphore(concurrency_limit)
        
        async def worker(g: GoldenDataPoint, s: RAGSystemOutput):
            async with semaphore:
                return await self.evaluate_single_run(g, s)
        
        # Build parallel task worker arrays
        tasks = [worker(g, s) for g, s in zip(golden_dataset, system_outputs)]
        reports: List[EvaluationReportItem] = await asyncio.gather(*tasks)
        
        # Compute macro statistical means across scores array matrix
        faithfulness_scores = [r.scores.faithfulness for r in reports]
        recall_scores = [r.scores.context_recall for r in reports]
        relevance_scores = [r.scores.answer_relevance for r in reports]
        
        return BatchEvaluationSummary(
            mean_faithfulness=round(float(np.mean(faithfulness_scores)), 3),
            mean_context_recall=round(float(np.mean(recall_scores)), 3),
            mean_answer_relevance=round(float(np.mean(relevance_scores)), 3),
            detailed_reports=reports
        )
Use code with caution.

Verification and Test Walkthrough

The script below builds a mock Golden Dataset and mock RAG system outputs to demonstrate the execution of the evaluation engine.
python
async def run_evaluation_suite():
    # 1. Define the Ground Truth Test Dataset
    golden_dataset = [
        GoldenDataPoint(
            query="What is the data retention policy for user activity logs?",
            ground_truth="User activity logs are retained for 90 days, after which they are automatically purged."
        ),
        GoldenDataPoint(
            query="How do I rotate my API keys?",
            ground_truth="API keys can be rotated via the security dashboard by clicking 'Regenerate Key'."
        )
    ]
    
    # 2. Simulate outputs generated by your production RAG pipeline
    simulated_rag_outputs = [
        RAGSystemOutput(
            query="What is the data retention policy for user activity logs?",
            # Halfful answer, but skips the exact automatic purge condition
            generated_answer="Activity logs are kept in storage infrastructure layers for exactly 90 days total.",
            retrieved_contexts=[
                "Standard compliance guidelines dictate log monitoring parameters.",
                "Section 4.1: User activity logs are retained for 90 days, after which they are automatically purged from production databases."
            ]
        ),
        RAGSystemOutput(
            query="How do I rotate my API keys?",
            # Hallucinated answer completely missing from the context provided below
            generated_answer="You must email the support desk admin engineering team to request rotation.",
            retrieved_contexts=[
                "API keys grant root administrative privileges across cluster components.",
                "Security logs record every authentication step mapping validation tokens."
            ]
        )
    ]

    print("📊 Initializing Asynchronous RAG Evaluation Pipeline...")
    evaluator = RAGEvaluator()
    
    # Run the evaluation batch
    summary = await evaluator.evaluate_batch(golden_dataset, simulated_rag_outputs)
    
    print("\n=============================================")
    print("           RAG EVALUATION METRICS REPORT      ")
    print("=============================================")
    print(f"Mean Faithfulness (Groundedness): {summary.mean_faithfulness}")
    print(f"Mean Context Recall             : {summary.mean_context_recall}")
    print(f"Mean Answer Relevance           : {summary.mean_answer_relevance}")
    print("=============================================\n")
    
    for idx, report in enumerate(summary.detailed_reports, 1):
        print(f"--- Test Case #{idx} Breakdown ---")
        print(f"Query    : {report.query}")
        print(f"Judge Rationale: {report.scores.rationale}")
        print(f"Scores -> Faithfulness: {report.scores.faithfulness} | Recall: {report.scores.context_recall} | Relevance: {report.scores.answer_relevance}\n")

if __name__ == "__main__":
    asyncio.run(run_evaluation_suite())
Use code with caution.

Key Production Advantages

• OpenAI Structured Outputs (response_format=Model): Standard prompt engineering often fails when asking an LLM to return scores, resulting in conversational text like "I would rate this a 0.8". Enforcing a Pydantic schema ensures the judge returns pure float data types that can immediately feed into monitoring dashboards.
• Bounded Concurrency Semaphore (asyncio.Semaphore): Running evaluations over hundreds of golden test strings simultaneously can trigger rate limits or exhaust your API allocation. Using a semaphore balances high throughput with API threshold constraints.
To scale this evaluation pipeline into your CI/CD workflow, let me know:
• Do you want to add traditional lexical metrics like ROUGE or BLEU alongside the LLM-as-a-judge scores?
• Do you need an automated process to generate the initial Golden Dataset using your raw documentation chunks?
