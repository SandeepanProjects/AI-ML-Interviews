# What is LLM-as-a-Judge?

**LLM-as-a-Judge** is an evaluation technique where one LLM evaluates the output of another LLM.

Instead of a human manually reviewing every answer:

```text
Prompt
  │
  ▼
Candidate LLM
  │
  ▼
Generated Answer
  │
  ▼
Judge LLM
  │
  ▼
Score / Verdict / Feedback
```

The judge LLM can evaluate:

* correctness
* relevance
* helpfulness
* instruction following
* groundedness
* hallucinations
* safety
* writing quality
* pairwise preference

---

# 1. Why do we need LLM-as-a-Judge?

Suppose you have:

```text
10,000 prompts
      ↓
10,000 model responses
```

Human evaluation of all responses is expensive and slow.

Automatic metrics such as BLEU or ROUGE have another problem: there can be **many correct answers**.

Example:

### Reference answer

```text
RAG retrieves relevant documents and uses them as context.
```

### Model answer A

```text
RAG searches for relevant information before the LLM generates an answer.
```

### Model answer B

```text
RAG uses retrieval to ground language-model responses in external knowledge.
```

Both may be good answers, but their word overlap with the reference can differ.

An LLM judge can reason about:

> Are these answers semantically correct and helpful?

---

# 2. Basic architecture

```text
                    Evaluation Dataset
                           │
                           ▼
                        Prompt
                           │
                           ▼
                  ┌─────────────────┐
                  │ Candidate Model │
                  └─────────────────┘
                           │
                           ▼
                    Generated Answer
                           │
                           ▼
                  ┌─────────────────┐
                  │    Judge LLM    │
                  └─────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Score / Verdict │
                  │ Explanation     │
                  └─────────────────┘
```

The important principle is:

```text
Candidate Model ≠ Judge Model
```

They can be different models, or sometimes the same model with a different role. Using a stronger and independent judge is generally preferable.

---

# 3. Example: evaluating correctness and helpfulness

Suppose the user asks:

```text
What is RAG?
```

Candidate answer:

```text
RAG retrieves relevant documents and provides them to an LLM as context before generating an answer.
```

We send the prompt and answer to the judge.

```python
JUDGE_PROMPT = """
You are an expert evaluator.

Evaluate the model response using the rubric below.

USER QUESTION:
{question}

MODEL RESPONSE:
{answer}

Evaluate:

1. Correctness
2. Relevance
3. Helpfulness
4. Clarity

Score each from 1 to 5.

Return JSON only:

{{
    "correctness": 0,
    "relevance": 0,
    "helpfulness": 0,
    "clarity": 0,
    "overall_score": 0,
    "reason": ""
}}
"""
```

Notice:

```text
You are a helpful assistant.
```

is not enough for a judge.

A good judge needs:

* a clear task
* explicit criteria
* a scoring scale
* structured output

---

# 4. Implement a judge with structured output

Use Pydantic to validate the judge output.

```python
from pydantic import BaseModel, Field


class JudgeResult(BaseModel):
    correctness: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    helpfulness: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    overall_score: float = Field(ge=1, le=5)
    reason: str
```

Now the evaluator function:

```python
import json


def judge_response(
    judge_client,
    question: str,
    answer: str
) -> JudgeResult:

    prompt = JUDGE_PROMPT.format(
        question=question,
        answer=answer
    )

    response = judge_client.generate(
        prompt=prompt,
        temperature=0
    )

    data = json.loads(response)

    return JudgeResult.model_validate(data)
```

Usage:

```python
result = judge_response(
    judge_client=judge_llm,
    question="What is RAG?",
    answer="""
    RAG retrieves relevant documents and provides
    them as context to an LLM before generating
    an answer.
    """
)

print(result)
```

Example result:

```text
correctness=5
relevance=5
helpfulness=5
clarity=5
overall_score=5.0
reason='The answer accurately explains the core RAG process.'
```

---

# 5. Why use `temperature=0`?

Evaluation should be as reproducible as possible.

Bad:

```text
Run 1 → Score 5
Run 2 → Score 3
Run 3 → Score 4
```

Better:

```python
temperature=0
```

This reduces randomness.

However:

> `temperature=0` does not guarantee perfect determinism.

Model-serving infrastructure, model versions, and decoding behavior can still introduce variation.

---

# 6. LLM-as-a-Judge for instruction following

Suppose the instruction is:

```text
Explain RAG in exactly three bullet points.
Do not mention LangChain.
```

Candidate answer:

```text
- RAG retrieves relevant information.
- It provides the information to an LLM.
- The LLM generates a grounded response.
```

Judge prompt:

```python
INSTRUCTION_JUDGE_PROMPT = """
You are an instruction-following evaluator.

Evaluate whether the model response followed
every instruction.

INSTRUCTION:
{instruction}

MODEL RESPONSE:
{response}

Check:

1. Required format
2. Number of bullet points
3. Required content
4. Forbidden content
5. Other explicit constraints

Return JSON only:

{{
    "followed_instruction": true,
    "score": 0,
    "violations": [],
    "reason": ""
}}
"""
```

Pydantic schema:

```python
from typing import List


class InstructionJudgeResult(BaseModel):

    followed_instruction: bool

    score: int = Field(
        ge=1,
        le=5
    )

    violations: List[str]

    reason: str
```

---

# 7. LLM-as-a-Judge for hallucination detection

This is especially useful for RAG.

Suppose the context is:

```text
Refunds take between 5 and 7 business days.
```

Model answer:

```text
Refunds usually arrive within 24 hours.
```

Judge prompt:

```python
GROUNDING_JUDGE_PROMPT = """
You are a factual grounding evaluator.

Determine whether the answer is supported
by the provided context.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
{answer}

Classify the answer as:

SUPPORTED:
The answer is fully supported by the context.

PARTIALLY_SUPPORTED:
Some claims are supported but some are not.

UNSUPPORTED:
The context does not support the answer.

CONTRADICTED:
The answer conflicts with the context.

Return JSON only:

{{
    "label": "",
    "hallucination_detected": false,
    "reason": ""
}}
"""
```

Schema:

```python
from typing import Literal


class GroundingJudgeResult(BaseModel):

    label: Literal[
        "SUPPORTED",
        "PARTIALLY_SUPPORTED",
        "UNSUPPORTED",
        "CONTRADICTED"
    ]

    hallucination_detected: bool

    reason: str
```

Function:

```python
def judge_grounding(
    judge_client,
    context: str,
    question: str,
    answer: str
) -> GroundingJudgeResult:

    prompt = GROUNDING_JUDGE_PROMPT.format(
        context=context,
        question=question,
        answer=answer
    )

    response = judge_client.generate(
        prompt=prompt,
        temperature=0
    )

    data = json.loads(response)

    return GroundingJudgeResult.model_validate(
        data
    )
```

Usage:

```python
result = judge_grounding(
    judge_client=judge_llm,
    context="""
    Refunds take between 5 and 7
    business days.
    """,
    question="How long does a refund take?",
    answer="Refunds usually arrive within 24 hours."
)

print(result)
```

Expected:

```text
label='CONTRADICTED'
hallucination_detected=True
```

---

# 8. Pointwise evaluation

The judge evaluates one response independently.

```text
Prompt
  │
  ├── Candidate Answer
  │
  ▼
Judge
  │
  ▼
Score 1–5
```

Example:

```python
POINTWISE_PROMPT = """
Rate this answer from 1 to 5.

QUESTION:
{question}

ANSWER:
{answer}

Rubric:

5 = Excellent
4 = Good
3 = Acceptable
2 = Poor
1 = Incorrect or unusable

Return JSON:

{{
    "score": 0,
    "reason": ""
}}
"""
```

Example output:

```json
{
    "score": 4,
    "reason": "Correct and relevant, but lacks examples."
}
```

---

# 9. Pairwise evaluation

Often, comparing two answers is more reliable.

Instead of:

```text
How good is this answer?
```

Ask:

```text
Which answer is better?
```

Architecture:

```text
                  Same Question
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
         Model A              Model B
             │                   │
             ▼                   ▼
         Answer A              Answer B
             │                   │
             └─────────┬─────────┘
                       ▼
                    Judge
                       │
             ┌─────────┼─────────┐
             ▼         ▼         ▼
             A wins   B wins    Tie
```

Judge prompt:

```python
PAIRWISE_PROMPT = """
You are an impartial evaluator.

Compare two responses to the same question.

QUESTION:
{question}

RESPONSE A:
{answer_a}

RESPONSE B:
{answer_b}

Evaluate:

1. Correctness
2. Relevance
3. Helpfulness
4. Clarity
5. Instruction following

Choose:

A
B
TIE

Return JSON only:

{{
    "winner": "",
    "reason": ""
}}
"""
```

Schema:

```python
class PairwiseJudgeResult(BaseModel):

    winner: Literal[
        "A",
        "B",
        "TIE"
    ]

    reason: str
```

Function:

```python
def judge_pair(
    judge_client,
    question,
    answer_a,
    answer_b
):

    prompt = PAIRWISE_PROMPT.format(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b
    )

    response = judge_client.generate(
        prompt=prompt,
        temperature=0
    )

    data = json.loads(response)

    return (
        PairwiseJudgeResult
        .model_validate(data)
    )
```

---

# 10. Position bias: an important problem

Suppose the judge sees:

```text
Response A = Better
Response B = Worse
```

The judge might develop a bias toward the first answer.

So evaluate twice:

### First

```text
A = Model 1
B = Model 2
```

### Then swap

```text
A = Model 2
B = Model 1
```

Code:

```python
def judge_pair_with_swap(
    judge_client,
    question,
    answer_a,
    answer_b
):

    result_1 = judge_pair(
        judge_client,
        question,
        answer_a,
        answer_b
    )

    result_2 = judge_pair(
        judge_client,
        question,
        answer_b,
        answer_a
    )

    return {
        "original_order": result_1,
        "swapped_order": result_2
    }
```

When interpreting the second result:

```text
Original order:

A = Model 1
B = Model 2

Swapped order:

A = Model 2
B = Model 1
```

You must map the winner back to the original model correctly.

A more robust implementation:

```python
def normalize_swapped_winner(
    winner: str
) -> str:

    mapping = {
        "A": "B",
        "B": "A",
        "TIE": "TIE"
    }

    return mapping[winner]
```

Then:

```python
def judge_pair_position_robust(
    judge_client,
    question,
    answer_a,
    answer_b
):

    first = judge_pair(
        judge_client,
        question,
        answer_a,
        answer_b
    )

    swapped = judge_pair(
        judge_client,
        question,
        answer_b,
        answer_a
    )

    normalized_second = (
        normalize_swapped_winner(
            swapped.winner
        )
    )

    if first.winner == normalized_second:
        final_winner = first.winner
    else:
        final_winner = "TIE"

    return {
        "winner": final_winner,
        "first_vote": first.winner,
        "second_vote": normalized_second
    }
```

This helps reduce position bias.

---

# 11. The biggest issue: LLM judge bias

LLM judges are not perfect.

Common biases include:

## A. Position bias

The first answer may be preferred.

## B. Verbosity bias

The judge may prefer:

```text
Long answer = Better
```

even when the long answer is unnecessary.

## C. Style bias

The judge may prefer polished language over factual correctness.

## D. Self-preference bias

A model may prefer answers generated by itself or similar models.

## E. Prompt sensitivity

Small changes in the evaluation prompt can change scores.

## F. Reference-answer bias

A judge may prefer answers similar to the reference even when another answer is equally correct.

---

# 12. How to reduce judge bias

## 1. Use explicit rubrics

Bad:

```text
Is this answer good?
```

Better:

```text
Score correctness independently.
Do not reward unnecessary length.
Penalize unsupported factual claims.
```

Example:

```python
ROBUST_JUDGE_PROMPT = """
You are an impartial evaluator.

Evaluate the answer strictly using this rubric.

CORRECTNESS:
Are factual claims correct?

RELEVANCE:
Does the answer directly address the question?

HELPFULNESS:
Does the answer provide useful information?

CLARITY:
Is the answer understandable?

IMPORTANT RULES:

- Do not reward unnecessary length.
- Do not prefer sophisticated vocabulary.
- Do not assume facts not present in the answer.
- Penalize factual errors.
- Penalize unsupported claims.
- Judge the answer, not its writing style alone.

QUESTION:
{question}

ANSWER:
{answer}

Return JSON only.
"""
```

---

## 2. Randomize answer order

```python
import random


def randomize_pair(
    answer_a,
    answer_b
):

    answers = [
        ("A", answer_a),
        ("B", answer_b)
    ]

    random.shuffle(answers)

    return answers
```

---

## 3. Use multiple judges

Instead of:

```text
One judge → final answer
```

Use:

```text
             ┌── Judge 1
Answer ──────┼── Judge 2
             └── Judge 3
                    │
                    ▼
                Aggregate
```

Code:

```python
def majority_vote(
    judge_results
):

    votes = {}

    for result in judge_results:

        winner = result.winner

        votes[winner] = (
            votes.get(winner, 0)
            + 1
        )

    return max(
        votes,
        key=votes.get
    )
```

---

# 13. Judge agreement

If you have multiple judges:

```text
Judge 1 → Score 5
Judge 2 → Score 4
Judge 3 → Score 5
```

Calculate average:

```python
def average_score(
    scores
):

    return (
        sum(scores)
        / len(scores)
    )
```

But also check disagreement:

```python
import statistics


def judge_variance(
    scores
):

    return statistics.pvariance(
        scores
    )
```

High variance means:

```text
Judges disagree
      ↓
Low confidence
      ↓
Possible human review
```

---

# 14. Add confidence

A production evaluator can return:

```python
class JudgeDecision(BaseModel):

    score: int = Field(
        ge=1,
        le=5
    )

    confidence: float = Field(
        ge=0,
        le=1
    )

    reason: str
```

Example:

```json
{
    "score": 2,
    "confidence": 0.95,
    "reason": "The answer directly contradicts the provided context."
}
```

Be careful: **a judge model's self-reported confidence is not necessarily calibrated**. It is useful metadata, not proof of reliability.

A stronger confidence signal can come from:

* agreement across multiple judges
* repeated evaluations
* agreement with deterministic checks
* agreement with human labels

---

# 15. A production-quality evaluator

Let's combine everything.

```python
from typing import Any
from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):

    correctness: int = Field(
        ge=1,
        le=5
    )

    relevance: int = Field(
        ge=1,
        le=5
    )

    helpfulness: int = Field(
        ge=1,
        le=5
    )

    groundedness: int = Field(
        ge=1,
        le=5
    )

    overall_score: float = Field(
        ge=1,
        le=5
    )

    hallucination_detected: bool

    reason: str
```

Evaluation function:

```python
def evaluate_with_llm_judge(
    judge_client,
    question: str,
    answer: str,
    context: str | None = None
) -> EvaluationResult:

    prompt = f"""
You are an expert and impartial evaluator.

Evaluate the candidate answer.

QUESTION:
{question}

CONTEXT:
{context or "No context provided"}

ANSWER:
{answer}

Score each dimension from 1 to 5.

CORRECTNESS:
Are the factual statements correct?

RELEVANCE:
Does the answer address the question?

HELPFULNESS:
Would this answer help the user?

GROUNDEDNESS:
If context is provided, is the answer supported by it?

HALLUCINATION:
Does the answer introduce unsupported factual claims?

IMPORTANT:
- Do not reward unnecessary verbosity.
- Penalize incorrect facts.
- Penalize contradictions with the context.
- If information is unavailable, do not assume it is true.
- Judge factual quality separately from writing style.

Return valid JSON only:

{{
    "correctness": 0,
    "relevance": 0,
    "helpfulness": 0,
    "groundedness": 0,
    "overall_score": 0,
    "hallucination_detected": false,
    "reason": ""
}}
"""

    raw_response = judge_client.generate(
        prompt=prompt,
        temperature=0
    )

    data = json.loads(
        raw_response
    )

    return EvaluationResult.model_validate(
        data
    )
```

Usage:

```python
result = evaluate_with_llm_judge(

    judge_client=judge_llm,

    question="""
    How long does a refund take?
    """,

    context="""
    Refunds take 5 to 7 business days.
    """,

    answer="""
    Refunds usually take 24 hours.
    """
)

print(result)
```

Expected:

```text
correctness=1
relevance=4
helpfulness=2
groundedness=1
overall_score=1.5
hallucination_detected=True
reason='The answer contradicts the provided refund policy.'
```

---

# 16. Batch evaluation

For a real project, you may evaluate thousands of examples.

```python
def evaluate_dataset(
    dataset,
    judge_client
):

    results = []

    for example in dataset:

        result = evaluate_with_llm_judge(

            judge_client=judge_client,

            question=example["question"],

            answer=example["answer"],

            context=example.get("context")
        )

        results.append({

            "id": example["id"],

            "evaluation":
                result.model_dump()
        })

    return results
```

Calculate metrics:

```python
def calculate_metrics(
    results
):

    total = len(results)

    average_score = sum(

        item["evaluation"][
            "overall_score"
        ]

        for item in results

    ) / total


    hallucinations = sum(

        item["evaluation"][
            "hallucination_detected"
        ]

        for item in results
    )


    return {

        "total_examples": total,

        "average_score":
            average_score,

        "hallucination_rate":
            hallucinations / total
    }
```

---

# 17. Compare two models

```text
                 Evaluation Dataset
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
      Model A                      Model B
          │                           │
          ▼                           ▼
      Answers A                    Answers B
          │                           │
          └─────────────┬─────────────┘
                        ▼
                   LLM Judge
                        │
                        ▼
                  Winner / Scores
```

Code:

```python
def compare_models_with_judge(
    dataset,
    model_a,
    model_b,
    tokenizer,
    judge_client
):

    results = []

    for example in dataset:

        answer_a = generate_response(
            model=model_a,
            tokenizer=tokenizer,
            prompt=example["prompt"]
        )

        answer_b = generate_response(
            model=model_b,
            tokenizer=tokenizer,
            prompt=example["prompt"]
        )

        judgment = judge_pair_position_robust(
            judge_client=judge_client,
            question=example["prompt"],
            answer_a=answer_a,
            answer_b=answer_b
        )

        results.append({

            "id": example["id"],

            "winner":
                judgment["winner"]
        })

    return results
```

Calculate win rate:

```python
def calculate_win_rates(
    results
):

    total = len(results)

    counts = {
        "A": 0,
        "B": 0,
        "TIE": 0
    }

    for result in results:

        counts[
            result["winner"]
        ] += 1

    return {

        "model_a_win_rate":
            counts["A"] / total,

        "model_b_win_rate":
            counts["B"] / total,

        "tie_rate":
            counts["TIE"] / total
    }
```

Example:

```text
Model A: 42%
Model B: 53%
Tie:      5%
```

Model B performs better according to the judge.

---

# 18. The best production architecture

For a real LLM or RAG application:

```text
                        Production Data
                              │
                              ▼
                     Sample Evaluation Set
                              │
                              ▼
                         Candidate Model
                              │
                              ▼
                           Responses
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
    Rule-based Eval       LLM Judge         Human Eval
          │                   │                   │
    JSON/schema            Semantic          Gold Standard
    Exact constraints      Quality
    Safety rules           Grounding
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
                       Evaluation Store
                              │
                              ▼
                          Dashboard
                              │
              ┌───────────────┼────────────────┐
              ▼               ▼                ▼
          Quality         Hallucination    Regression
```

### Important principle

Do **not** rely only on an LLM judge.

Use:

```text
Rule-based evaluation
        +
LLM-as-a-Judge
        +
Human evaluation
```

---

# 19. LLM-as-a-Judge vs Human Evaluation

| Feature             | LLM Judge                       | Human Evaluation                 |
| ------------------- | ------------------------------- | -------------------------------- |
| Speed               | Very fast                       | Slow                             |
| Cost                | Lower at scale                  | Expensive                        |
| Scale               | Thousands/millions              | Limited                          |
| Semantic evaluation | Good                            | Excellent                        |
| Domain expertise    | Depends on judge                | Can use experts                  |
| Consistency         | Usually high with fixed prompts | Variable                         |
| Gold standard       | No                              | Yes                              |
| Best use            | Continuous evaluation           | Calibration and final validation |

---

# 20. Best practice: calibrate against humans

The most important production workflow is:

```text
Step 1
Humans label 500–1000 examples
        │
        ▼
Step 2
Run LLM judge on same examples
        │
        ▼
Step 3
Compare LLM judge vs humans
        │
        ▼
Step 4
Measure agreement
        │
        ▼
Step 5
Improve judge prompt/rubric
        │
        ▼
Step 6
Use calibrated judge for large-scale evaluation
        │
        ▼
Step 7
Periodically validate again with humans
```

Example agreement code:

```python
from sklearn.metrics import (
    cohen_kappa_score
)


human_scores = [
    5, 4, 3, 5, 2
]

judge_scores = [
    5, 4, 3, 4, 2
]


agreement = cohen_kappa_score(
    human_scores,
    judge_scores,
    weights="quadratic"
)

print(
    f"Human-Judge Agreement: "
    f"{agreement:.3f}"
)
```

If agreement is poor:

```text
Human says → Good
LLM judge says → Bad
```

you should **not trust the judge for that task yet**.

---

# Interview-ready answer

> **LLM-as-a-Judge is an evaluation technique where an LLM evaluates another model's response using a predefined rubric. It is useful for open-ended tasks where traditional metrics like BLEU and ROUGE are insufficient because multiple responses can be valid.**
>
> **I provide the judge with the original prompt, optionally the reference answer or retrieved context, and the candidate response. The judge evaluates dimensions such as correctness, relevance, helpfulness, instruction following, groundedness, and hallucination. I use structured output such as Pydantic schemas, deterministic settings, explicit rubrics, and validation.**
>
> **For comparing models, I often use pairwise evaluation and randomize answer order to reduce position bias. I also watch for verbosity bias and self-preference bias. In production, I don't rely solely on the judge—I combine rule-based evaluation, LLM judging, and periodic human evaluation. Most importantly, I calibrate the LLM judge against a human-labeled gold dataset before trusting it at scale.**

## Key takeaway

```text
LLM-as-a-Judge is not:

"Ask an LLM if another answer is good."

It is:

Clear evaluation task
        +
Detailed rubric
        +
Structured output
        +
Bias mitigation
        +
Human calibration
        +
Large-scale automated evaluation
```

That makes LLM-as-a-Judge a powerful component of a **production LLM evaluation pipeline**.
