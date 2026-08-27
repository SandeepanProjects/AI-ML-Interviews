# How Do You Evaluate Instruction Following in an LLM?

Instruction following means evaluating whether an LLM **actually obeys the requirements in a prompt**, not merely whether its answer is fluent.

For example:

```text
Instruction:
Return exactly 3 bullet points.
Do not mention Python.

Good output:
- Use caching.
- Add retries.
- Monitor latency.

Bad output:
Here are some suggestions:
- Use Python caching.
- Add retries.
- Monitor latency.
```

The bad output may be useful, but it **failed the instruction** because it mentioned Python.

---

# 1. What should we evaluate?

A good instruction-following evaluation usually checks multiple dimensions:

```text
                    Prompt
                      │
                      ▼
                LLM Response
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   Constraint      Correctness    Quality
   Following
        │
        ├── Format
        ├── Length
        ├── Required content
        ├── Forbidden content
        ├── Style
        └── Multi-step instructions
```

For example:

```text
Instruction:
Explain RAG in exactly 3 bullet points,
each bullet must contain fewer than 15 words,
and do not mention LangChain.
```

We need to check:

1. Exactly 3 bullets?
2. Each bullet < 15 words?
3. Does it avoid "LangChain"?
4. Is the explanation actually about RAG?
5. Is the response useful?

---

# 2. Build an instruction-following evaluation dataset

Instead of only storing a reference answer, store the **constraints**.

```python
evaluation_dataset = [
    {
        "id": "format_001",

        "prompt": """
Explain RAG in exactly 3 bullet points.
Do not mention LangChain.
""",

        "constraints": {
            "exact_bullet_count": 3,
            "forbidden_words": [
                "langchain"
            ]
        }
    },

    {
        "id": "format_002",

        "prompt": """
Return a JSON object with exactly these fields:
name, age, city.
""",

        "constraints": {
            "output_format": "json",

            "required_keys": [
                "name",
                "age",
                "city"
            ],

            "no_extra_keys": True
        }
    }
]
```

This is better than only:

```text
Prompt
Reference Answer
```

because instruction-following often has **many valid answers**.

---

# 3. Generate deterministic outputs

For evaluation, use deterministic generation.

```python
import torch


def generate_response(
    model,
    tokenizer,
    prompt,
    max_new_tokens=300
):

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )

    device = (
        model.get_input_embeddings()
        .weight
        .device
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[
        0,
        inputs["input_ids"].shape[1]:
    ]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )
```

Why:

```python
do_sample=False
```

Because evaluation should ideally be reproducible.

---

# 4. Evaluate exact bullet count

Suppose the instruction says:

```text
Return exactly 3 bullet points.
```

We can write a deterministic evaluator.

```python
import re


def count_bullets(text):

    lines = text.splitlines()

    bullet_pattern = re.compile(
        r"^\s*[-*•]\s+"
    )

    bullets = [
        line
        for line in lines
        if bullet_pattern.match(line)
    ]

    return len(bullets)
```

Test:

```python
response = """
- RAG retrieves relevant information.
- The LLM uses retrieved context.
- This improves factual accuracy.
"""


count = count_bullets(response)

print(count)
```

Output:

```text
3
```

Evaluator:

```python
def evaluate_bullet_count(
    response,
    expected_count
):

    actual_count = count_bullets(
        response
    )

    return {
        "passed": (
            actual_count == expected_count
        ),

        "expected": expected_count,

        "actual": actual_count
    }
```

---

# 5. Evaluate forbidden words

Instruction:

```text
Do not mention LangChain.
```

Code:

```python
def check_forbidden_words(
    response,
    forbidden_words
):

    response_lower = response.lower()

    found = []

    for word in forbidden_words:

        if word.lower() in response_lower:

            found.append(word)

    return {
        "passed": len(found) == 0,
        "violations": found
    }
```

Example:

```python
response = """
- RAG retrieves relevant documents.
- LangChain can help build pipelines.
- The LLM generates an answer.
"""


result = check_forbidden_words(
    response,
    ["LangChain"]
)

print(result)
```

Output:

```python
{
    "passed": False,
    "violations": ["LangChain"]
}
```

---

# 6. Evaluate required words or concepts

Instruction:

```text
Explain RAG and mention embeddings.
```

```python
def check_required_words(
    response,
    required_words
):

    response_lower = response.lower()

    missing = []

    for word in required_words:

        if word.lower() not in response_lower:

            missing.append(word)

    return {
        "passed": len(missing) == 0,
        "missing": missing
    }
```

Example:

```python
response = """
RAG retrieves relevant documents before generating an answer.
Embeddings help find semantically similar documents.
"""


result = check_required_words(
    response,
    [
        "RAG",
        "embeddings"
    ]
)

print(result)
```

---

# 7. Evaluate word limits

Instruction:

```text
Answer in fewer than 50 words.
```

Code:

```python
def count_words(text):

    return len(
        text.split()
    )


def check_max_words(
    response,
    max_words
):

    word_count = count_words(
        response
    )

    return {
        "passed": (
            word_count <= max_words
        ),

        "max_words": max_words,

        "actual_words": word_count
    }
```

Usage:

```python
result = check_max_words(
    response,
    max_words=50
)

print(result)
```

---

# 8. Evaluate JSON instruction following

Instruction:

```text
Return valid JSON with:
name, age, city
```

## Step 1: Check valid JSON

```python
import json


def parse_json(
    response
):

    try:

        data = json.loads(
            response
        )

        return {
            "valid": True,
            "data": data
        }

    except json.JSONDecodeError as error:

        return {
            "valid": False,
            "error": str(error)
        }
```

## Step 2: Check required keys

```python
def check_json_keys(
    data,
    required_keys,
    no_extra_keys=False
):

    actual_keys = set(
        data.keys()
    )

    required_keys = set(
        required_keys
    )

    missing = (
        required_keys
        - actual_keys
    )

    extra = (
        actual_keys
        - required_keys
    )

    passed = (
        len(missing) == 0
    )

    if no_extra_keys:
        passed = (
            passed
            and len(extra) == 0
        )

    return {
        "passed": passed,

        "missing": list(missing),

        "extra": list(extra)
    }
```

Full evaluator:

```python
def evaluate_json_instruction(
    response,
    required_keys,
    no_extra_keys=False
):

    parsed = parse_json(
        response
    )

    if not parsed["valid"]:

        return {
            "passed": False,
            "reason": "Invalid JSON"
        }

    if not isinstance(
        parsed["data"],
        dict
    ):

        return {
            "passed": False,
            "reason": "Output is not JSON object"
        }

    return check_json_keys(
        parsed["data"],
        required_keys,
        no_extra_keys
    )
```

Example:

```python
response = """
{
    "name": "John",
    "age": 30,
    "city": "Bangalore"
}
"""


result = evaluate_json_instruction(
    response=response,

    required_keys=[
        "name",
        "age",
        "city"
    ],

    no_extra_keys=True
)

print(result)
```

---

# 9. Use Pydantic for stronger validation

For production applications, this is better.

```python
from pydantic import BaseModel
import json


class Person(BaseModel):

    name: str

    age: int

    city: str
```

Evaluate:

```python
def validate_person(
    response
):

    try:

        data = json.loads(
            response
        )

        person = Person.model_validate(
            data
        )

        return {
            "passed": True,
            "data": person
        }

    except Exception as error:

        return {
            "passed": False,
            "error": str(error)
        }
```

This checks:

```text
✓ Valid JSON
✓ Required fields
✓ Data types
✓ Schema
```

---

# 10. Evaluate multi-constraint instructions

Suppose:

```text
Explain RAG using exactly 3 bullet points.
Each bullet must have fewer than 12 words.
Do not mention LangChain.
```

Create a combined evaluator.

```python
def get_bullet_texts(
    response
):

    lines = response.splitlines()

    pattern = re.compile(
        r"^\s*[-*•]\s+(.+)"
    )

    bullets = []

    for line in lines:

        match = pattern.match(
            line
        )

        if match:

            bullets.append(
                match.group(1)
            )

    return bullets
```

Evaluator:

```python
def evaluate_rag_instruction(
    response
):

    bullets = get_bullet_texts(
        response
    )


    # ----------------------------------------
    # 1. Exactly 3 bullets
    # ----------------------------------------

    bullet_count_passed = (
        len(bullets) == 3
    )


    # ----------------------------------------
    # 2. Each bullet < 12 words
    # ----------------------------------------

    word_counts = [
        len(bullet.split())
        for bullet in bullets
    ]

    length_passed = all(
        count < 12
        for count in word_counts
    )


    # ----------------------------------------
    # 3. Forbidden word
    # ----------------------------------------

    forbidden_passed = (
        "langchain"
        not in response.lower()
    )


    overall_passed = (
        bullet_count_passed
        and length_passed
        and forbidden_passed
    )


    return {

        "overall_passed": overall_passed,

        "bullet_count": {
            "passed": bullet_count_passed,
            "actual": len(bullets)
        },

        "word_limit": {
            "passed": length_passed,
            "counts": word_counts
        },

        "forbidden_words": {
            "passed": forbidden_passed
        }
    }
```

Example:

```python
response = """
- RAG retrieves relevant documents before generating an answer.
- Retrieved context helps improve answer grounding and accuracy.
- The language model generates responses using that context.
"""


result = evaluate_rag_instruction(
    response
)

print(result)
```

This is **constraint-based instruction evaluation**.

---

# 11. Build a generic rule-based evaluator

Now let's make this reusable.

```python
def evaluate_constraints(
    response,
    constraints
):

    results = {}

    # ---------------------------------------
    # Bullet count
    # ---------------------------------------

    if "exact_bullet_count" in constraints:

        result = evaluate_bullet_count(
            response,
            constraints[
                "exact_bullet_count"
            ]
        )

        results[
            "bullet_count"
        ] = result


    # ---------------------------------------
    # Forbidden words
    # ---------------------------------------

    if "forbidden_words" in constraints:

        result = check_forbidden_words(
            response,
            constraints[
                "forbidden_words"
            ]
        )

        results[
            "forbidden_words"
        ] = result


    # ---------------------------------------
    # Required words
    # ---------------------------------------

    if "required_words" in constraints:

        result = check_required_words(
            response,
            constraints[
                "required_words"
            ]
        )

        results[
            "required_words"
        ] = result


    # ---------------------------------------
    # Maximum words
    # ---------------------------------------

    if "max_words" in constraints:

        result = check_max_words(
            response,
            constraints[
                "max_words"
            ]
        )

        results[
            "max_words"
        ] = result


    # ---------------------------------------
    # Overall score
    # ---------------------------------------

    passed = all(
        item["passed"]
        for item in results.values()
    )


    results[
        "overall_passed"
    ] = passed


    return results
```

Usage:

```python
constraints = {

    "exact_bullet_count": 3,

    "forbidden_words": [
        "langchain"
    ],

    "max_words": 100
}


response = """
- RAG retrieves relevant information.
- The information is provided to the language model.
- The model generates a grounded response.
"""


result = evaluate_constraints(
    response,
    constraints
)

print(result)
```

---

# 12. Measure instruction-following accuracy

Suppose we have 100 test prompts.

```python
def evaluate_dataset(
    model,
    tokenizer,
    evaluation_dataset
):

    results = []

    for example in evaluation_dataset:

        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=example["prompt"]
        )


        evaluation = evaluate_constraints(
            response,
            example["constraints"]
        )


        results.append(
            {
                "id": example["id"],

                "prompt": example["prompt"],

                "response": response,

                "evaluation": evaluation
            }
        )


    return results
```

Calculate pass rate:

```python
def calculate_instruction_following_rate(
    results
):

    passed = sum(
        item["evaluation"][
            "overall_passed"
        ]
        for item in results
    )


    return (
        passed
        / len(results)
    )
```

Example:

```python
results = evaluate_dataset(
    model,
    tokenizer,
    evaluation_dataset
)


score = (
    calculate_instruction_following_rate(
        results
    )
)


print(
    f"Instruction Following Rate: "
    f"{score:.2%}"
)
```

Output:

```text
Instruction Following Rate: 87.00%
```

This is a useful metric:

$$
Instruction\ Following\ Rate =
\frac{Instructions\ Fully\ Followed}
{Total\ Instructions}
$$

---

# 13. Constraint-level metrics

Do not only calculate overall pass/fail.

For example:

```text
Overall pass rate: 80%

But:

JSON format:       98%
Length constraint:  95%
Forbidden words:    92%
Exact bullet count: 72%
```

This tells you what the model struggles with.

Code:

```python
def calculate_constraint_metrics(
    results
):

    metrics = {
        "bullet_count": [],
        "forbidden_words": [],
        "required_words": [],
        "max_words": []
    }


    for result in results:

        evaluation = result[
            "evaluation"
        ]

        for metric_name in metrics:

            if metric_name in evaluation:

                metrics[
                    metric_name
                ].append(

                    evaluation[
                        metric_name
                    ]["passed"]

                )


    return {

        metric_name: (
            sum(values)
            / len(values)
            if values
            else None
        )

        for metric_name, values
        in metrics.items()
    }
```

---

# 14. Evaluate instruction difficulty

You should categorize your test set.

```text
Level 1:
Single instruction

"Answer in JSON."

Level 2:
Multiple instructions

"Answer in JSON.
Include name and age.
No extra fields."

Level 3:
Complex constraints

"Answer in JSON.
Use only these fields.
Sort results.
Do not mention internal data.
Keep response under 100 words."
```

Example dataset:

```python
evaluation_dataset = [

    {
        "id": "easy_001",

        "difficulty": "easy",

        "prompt": "Answer in exactly 2 sentences.",

        "constraints": {
            "exact_sentence_count": 2
        }
    },

    {
        "id": "medium_001",

        "difficulty": "medium",

        "prompt": """
Return exactly 3 bullet points.
Do not mention Python.
""",

        "constraints": {
            "exact_bullet_count": 3,

            "forbidden_words": [
                "python"
            ]
        }
    }
]
```

Then calculate performance by difficulty.

```text
Easy:   98%
Medium: 87%
Hard:   61%
```

This is much more useful than one overall number.

---

# 15. LLM-as-a-Judge for semantic instructions

Rule-based evaluation cannot handle everything.

Example:

```text
Instruction:
Explain RAG to a beginner using a simple analogy.
```

How do you programmatically check:

```text
"Is this analogy understandable?"
```

This is subjective.

Use an LLM judge.

---

## Judge prompt

```python
def build_judge_prompt(
    instruction,
    response
):

    return f"""
You are an expert evaluator.

Evaluate whether the model followed the instruction.

INSTRUCTION:
{instruction}

MODEL RESPONSE:
{response}

Evaluate these criteria:

1. Did it follow all explicit constraints?
2. Did it satisfy the requested format?
3. Did it provide the requested content?
4. Did it violate any restrictions?
5. Did it follow implicit requirements?

Return JSON only:

{{
    "instruction_following_score": 0,
    "followed_instruction": false,
    "violations": [],
    "reason": ""
}}

Score from 1 to 5.
"""
```

Example judge output:

```json
{
    "instruction_following_score": 4,
    "followed_instruction": true,
    "violations": [],
    "reason": "The response followed the requested format and used a simple analogy."
}
```

---

# 16. Implement an LLM judge

Below is a provider-agnostic pattern:

```python
import json


def evaluate_with_judge(
    judge_client,
    instruction,
    response
):

    prompt = build_judge_prompt(
        instruction,
        response
    )


    judge_response = judge_client.generate(
        prompt,
        temperature=0
    )


    try:

        result = json.loads(
            judge_response
        )

        return result

    except json.JSONDecodeError:

        return {
            "instruction_following_score": None,

            "followed_instruction": False,

            "violations": [
                "Judge returned invalid JSON"
            ],

            "reason": judge_response
        }
```

For evaluation:

```text
temperature = 0
```

or equivalent deterministic settings, to reduce evaluator variability.

---

# 17. Rule-based + LLM judge is the best approach

In a production evaluation pipeline:

```text
                  Generated Answer
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼

    Deterministic Checks             LLM Judge
          │                             │

     JSON valid?                    Semantic intent?
     3 bullets?                     Tone correct?
     < 100 words?                   Helpful?
     No forbidden words?            Actually followed?
          │                             │
          └──────────────┬──────────────┘
                         ▼
                   Final Evaluation
```

Example:

```python
def evaluate_instruction(
    instruction,
    response,
    constraints,
    judge_client=None
):

    rule_results = (
        evaluate_constraints(
            response,
            constraints
        )
    )


    result = {
        "rule_based": rule_results
    }


    if judge_client:

        judge_result = (
            evaluate_with_judge(
                judge_client,
                instruction,
                response
            )
        )

        result[
            "llm_judge"
        ] = judge_result


    return result
```

---

# 18. Compare the base model vs fine-tuned model

This is extremely important after instruction tuning.

Use the **same frozen evaluation dataset**.

```python
def compare_models(
    base_model,
    fine_tuned_model,
    tokenizer,
    dataset
):

    report = []

    for example in dataset:

        prompt = example["prompt"]

        constraints = (
            example["constraints"]
        )


        # Base model
        base_response = (
            generate_response(
                base_model,
                tokenizer,
                prompt
            )
        )


        base_result = (
            evaluate_constraints(
                base_response,
                constraints
            )
        )


        # Fine-tuned model
        ft_response = (
            generate_response(
                fine_tuned_model,
                tokenizer,
                prompt
            )
        )


        ft_result = (
            evaluate_constraints(
                ft_response,
                constraints
            )
        )


        report.append({

            "id": example["id"],

            "base_passed":
                base_result[
                    "overall_passed"
                ],

            "fine_tuned_passed":
                ft_result[
                    "overall_passed"
                ]
        })


    return report
```

Calculate:

```text
Base Model Instruction Following:       72%
Fine-Tuned Model Instruction Following: 91%
```

That gives you direct evidence that fine-tuning worked.

---

# 19. Important instruction-following metrics

| Metric                     | Meaning                                        |
| -------------------------- | ---------------------------------------------- |
| Instruction Following Rate | % examples where all constraints are satisfied |
| Constraint Pass Rate       | % individual constraints satisfied             |
| Format Accuracy            | Correct output format                          |
| JSON Validity              | % valid JSON outputs                           |
| Schema Validity            | % outputs matching required schema             |
| Length Compliance          | % outputs within limits                        |
| Forbidden Content Rate     | % outputs avoiding forbidden content           |
| Multi-Constraint Accuracy  | % complex prompts fully followed               |
| Judge Score                | Semantic instruction-following quality         |
| Regression Rate            | Previously passing instructions that now fail  |

---

# 20. A complete production-style evaluator

```python
from dataclasses import dataclass
from typing import Any


@dataclass
class EvaluationResult:

    example_id: str

    response: str

    overall_passed: bool

    details: dict[str, Any]


def run_instruction_evaluation(
    model,
    tokenizer,
    dataset
):

    results = []


    for example in dataset:

        # ----------------------------------
        # Generate
        # ----------------------------------

        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=example["prompt"]
        )


        # ----------------------------------
        # Evaluate constraints
        # ----------------------------------

        evaluation = (
            evaluate_constraints(
                response=response,
                constraints=example[
                    "constraints"
                ]
            )
        )


        # ----------------------------------
        # Store result
        # ----------------------------------

        results.append(

            EvaluationResult(

                example_id=example["id"],

                response=response,

                overall_passed=evaluation[
                    "overall_passed"
                ],

                details=evaluation
            )
        )


    return results
```

Summary:

```python
def create_summary(
    results
):

    total = len(results)

    passed = sum(
        result.overall_passed
        for result in results
    )


    return {

        "total_examples": total,

        "passed": passed,

        "failed": total - passed,

        "instruction_following_rate": (
            passed / total
            if total > 0
            else 0
        )
    }
```

Usage:

```python
results = run_instruction_evaluation(
    model=model,
    tokenizer=tokenizer,
    dataset=evaluation_dataset
)


summary = create_summary(
    results
)


print(summary)
```

Example:

```text
{
    'total_examples': 100,
    'passed': 89,
    'failed': 11,
    'instruction_following_rate': 0.89
}
```

---

# Interview-ready answer

> **To evaluate instruction following, I don't rely only on BLEU or ROUGE because there can be many valid answers. I create a held-out instruction evaluation dataset where each prompt includes explicit constraints such as required format, length limits, required information, forbidden content, and multi-step requirements.**
>
> **For deterministic constraints, I use programmatic evaluators—for example, checking JSON schema validity with Pydantic, exact bullet count with regex, word limits, required fields, and forbidden terms. I calculate an instruction-following rate based on the percentage of examples where all constraints are satisfied.**
>
> **For semantic or subjective instructions, such as “explain simply” or “use a helpful analogy,” I use an LLM-as-a-judge with a fixed rubric and calibrate it against human evaluation. Finally, I compare the base model and fine-tuned model on the same frozen evaluation set and measure performance separately for simple, multi-step, adversarial, and regression instructions.**

## The key principle

```text
Fine-tuning evaluation
       ≠
Only "Does the answer match the reference?"

Instruction-following evaluation
       =
"Did the model satisfy every requirement in the instruction?"
```

That distinction is particularly important for evaluating **instruction-tuned LLMs**.
