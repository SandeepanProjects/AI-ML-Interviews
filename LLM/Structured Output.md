# Structured Output in LLMs: How to Force Valid JSON in Production

Structured output means making an LLM return data in a predefined format, such as JSON, rather than unrestricted natural-language text.

For production AI systems, this is essential when an LLM must communicate with FastAPI endpoints, databases, other agents, workflow engines, or frontend applications.

For example, imagine your financial compliance AI reviews this statement:

> “Our investment strategy guarantees a 20% annual return.”

You don't want the LLM to return a long paragraph that your application must interpret. You want a predictable object that your backend can validate and process.

Example structured response

```

{
  "risk_level": "high",
  "is_compliant": false,
  "findings": [
    {
      "claim": "guarantees a 20% annual return",
      "reason": "The guarantee requires authoritative supporting evidence and compliance review.",
      "policy_reference": null
    }
  ],
  "requires_human_review": true
}
  
```

Illustrative output only. The actual compliance decision must depend on verified policies and evidence.

Your backend can now read `risk_level`, inspect `findings`, and route high-risk reviews to a human reviewer without trying to extract information from prose.

One important distinction: valid JSON is not necessarily correct JSON semantically. A response can be syntactically valid and still contain a false claim, an invalid policy reference, or an incorrect risk level.

Let's work through how to generate, enforce, and validate structured output properly.

## 1. What problems does structured output solve?

Consider three possible LLM responses to the same request.

Unstructured text

The statement appears risky because it promises returns. A compliance reviewer should investigate it.

A human can understand it, but application code must infer the fields.

JSON-looking text

```

Here is the result:
{"risk_level": "high", "is_compliant": false}
  
```

The explanation before the object may break code that expects the entire response to be JSON.

Structured JSON

```

{
  "risk_level": "high",
  "is_compliant": false
}
  
```

The response can be parsed, validated, and passed to downstream services.

Structured output is especially useful for:

- Agent-to-agent communication.
- Tool calling and workflow routing.
- Classification and extraction.
- API responses.
- Compliance, risk, and fraud review.
- Database persistence and reporting.

## 2. Four ways to get JSON from an LLM

These approaches are related, but they do not provide the same level of reliability.

Method 1: Prompting

Ask the model to return only JSON. Easy to implement, but the model may still produce invalid JSON or extra text.

Method 2: Schema validation

Parse the output and validate it using Pydantic or JSON Schema. This catches errors but doesn't, by itself, force the model to produce valid JSON.

Method 3: Constrained decoding / native structured output

Use a model API or inference engine that supports a JSON Schema or constrained-output mode. This is stronger than prompting alone.

Method 4: Validation plus controlled retry

Validate the result, retry transient or repairable failures within a strict limit, and reject or escalate anything that remains invalid.

For a production system, I recommend combining native structured output when available, Pydantic validation, and explicit failure handling.

## 3. Method 1 — Ask the LLM to return JSON using a prompt

This is the simplest approach.

```
import jsonprompt = """Analyze this claim:"Our investment strategy guarantees a 20% annual return."Return ONLY a JSON object with these keys:- risk_level: low, medium, or high- is_compliant: boolean- reason: stringDo not include Markdown or any text outside the JSON."""# Assume llm.generate(prompt) returns the model's text.response = llm.generate(prompt)result = json.loads(response)print(result["risk_level"])print(result["is_compliant"])
```

This is conceptual code; `llm.generate` depends on your model provider or inference library.

### Why this is not enough

The model might return:

```
Here is the JSON:
{"risk_level": "high", "is_compliant": "false"}
```

There are two issues:

1. The extra text makes the full response invalid JSON.
2. `"false"` is a string, not the JSON boolean `false`.

`json.loads()` catches the first problem if you pass the entire response. It won't catch the second because `"false"` is valid JSON—it just has the wrong type for your application.

Lesson: prompting expresses your desired format; it does not guarantee compliance with that format.

## 4. Method 2 — Validate JSON with Pydantic

Pydantic is a Python library that validates data against typed models. It is widely useful in FastAPI applications.

Install it:

```
pip install pydantic
```

Let's define a strict schema for the compliance-review response.

```
from typing import Literalfrom pydantic import BaseModel, ConfigDict, Fieldclass ComplianceFinding(BaseModel):    model_config = ConfigDict(extra="forbid", strict=True)    claim: str = Field(min_length=1)    reason: str = Field(min_length=1)    policy_reference: str | Noneclass ComplianceReview(BaseModel):    model_config = ConfigDict(extra="forbid", strict=True)    risk_level: Literal["low", "medium", "high"]    is_compliant: bool    findings: list[ComplianceFinding]    requires_human_review: boolraw_json = """{  "risk_level": "high",  "is_compliant": false,  "findings": [    {      "claim": "guarantees a 20% annual return",      "reason": "The claim needs authoritative evidence and review.",      "policy_reference": null    }  ],  "requires_human_review": true}"""
```

### What does Pydantic validate?

- `Literal["low", "medium", "high"]` restricts the risk-level value.
- `bool` requires a boolean when strict validation is enabled.
- `list[ComplianceFinding]` requires each finding to follow its own schema.
- `str | None` permits a string or `null` for the policy reference.
- `extra="forbid"` rejects unexpected fields.
- `Field(min_length=1)` rejects empty text for required descriptions.

Because `strict=True` is enabled, JSON types must match the expected Python types. For example, `"is_compliant": "false"` will be rejected rather than silently treated as a boolean.

### What Pydantic does not validate

Pydantic will not know whether the claim actually violates a financial policy. It will not verify whether a citation exists in your policy database, either.

You need separate business rules and evidence checks for those decisions.

## 5. Method 3 — Use native structured output

When a model provider supports structured outputs, send a schema to the API rather than merely requesting JSON in natural language.

Here's a concrete example using the OpenAI Python SDK with a Pydantic model.

Install dependencies:

```
pip install openai pydantic
```

Set your API key in the environment:

```
export OPENAI_API_KEY="your-api-key"
```

Do not hardcode or commit your real key into source control.

```
from typing import Literalfrom pydantic import BaseModel, ConfigDict, Fieldfrom openai import OpenAIclass Finding(BaseModel):    model_config = ConfigDict(extra="forbid")    claim: str    reason: str    policy_reference: str | Noneclass ReviewResult(BaseModel):    model_config = ConfigDict(extra="forbid")    risk_level: Literal["low", "medium", "high"]    is_compliant: bool    findings: list[Finding]    requires_human_review: boolclient = OpenAI()response = client.responses.parse(    model="gpt-4.1-mini",    input=[        {            "role": "system",            "content": (                "Review financial marketing claims. "                "Use only the evidence provided. "                "Do not invent policy references. "                "If evidence is insufficient, state that "                "in the findings and request human review."
```

This example uses the OpenAI Responses API's parsing interface and a supported model. Provider SDKs and model capabilities can change, so verify the installed SDK documentation and model compatibility for your deployment.

### Why is this better?

The schema is provided to the model API as a structured-output constraint. Instead of relying only on a textual instruction, the API uses its supported structured-output mechanism to constrain the response format.

However, native structured output is not a guarantee that:

- The model's factual claims are correct.
- A policy citation is real.
- The model followed every business rule.
- The request cannot be refused or interrupted.
- The output meets every application-specific invariant.

Your application must still handle refusal, incomplete output, API errors, and semantic validation.

## 6. How constrained decoding forces JSON

Let's go one level deeper into how structured generation can work.

During ordinary generation, an LLM predicts a probability distribution over the next token. The decoder selects a token, appends it to the output, and repeats.

With constrained decoding, the generation system restricts which tokens can be selected at each step so the output follows a grammar or schema.

For example, if the schema requires:

```
{
  "risk_level": "low"
}
```

the allowed values for `risk_level` may be restricted to `"low"`, `"medium"`, or `"high"`.

Conceptually:

LLM predicts next-token scores

Schema / grammar constraint

Restrict token choices to those that can continue a valid output.

Select an allowed token

Repeat until the structured response is complete or generation stops.

The exact mechanism varies by provider and inference engine. Some APIs use constrained decoding internally; others provide different structured-output guarantees.

Important: a schema that accepts `"risk_level": "high"` cannot determine whether `"high"` is the correct answer. Syntax and type constraints are not a substitute for domain reasoning and validation.

## 7. Method 4 — Production validation, retry, and safe failure

In a production system, never assume that a model call will always return a usable response. It can fail due to timeouts, refusals, truncation, schema incompatibility, or unexpected provider errors.

A robust design has three layers:

1. Format enforcement: native structured output or constrained decoding.
2. Data validation: parse the response and validate it with Pydantic.
3. Business validation: verify claims, policy versions, citations, permissions, and workflow rules.

Here is a reusable validation-and-retry pattern for a provider that returns raw JSON text. It is a fallback example; if your API already enforces a schema, use that capability as well.

```
import jsonfrom pydantic import ValidationErrorclass StructuredOutputError(Exception):    passdef generate_validated_json(    llm,    prompt: str,    schema_model,    max_attempts: int = 2,):    if max_attempts < 1:        raise ValueError("max_attempts must be at least 1")    last_error = None    for attempt in range(max_attempts):        try:            raw_text = llm.generate(prompt)            # 1. Require the entire response to be valid JSON.            data = json.loads(raw_text)            # 2. Validate the expected data structure and types.            result = schema_model.model_validate(data)            # 3. Return only validated data.            return result        except (json.JSONDecodeError, ValidationError) as exc:            last_error = exc
```

This example retries only parsing and schema-validation failures. In a real system, keep the original prompt immutable, pass validation errors as bounded repair feedback, and distinguish retryable provider failures from permanent errors. Do not retry indefinitely, and do not treat a successful parse as approval of the business decision.

For compliance workflows, an exhausted retry budget should result in a controlled failure or a review queue—not a fabricated default answer.

## 8. Integrate structured output into FastAPI

Now let's connect the model output to an API.

The following example focuses on the response contract. `run_model_review` represents your model-service layer, which should call your provider with structured-output constraints and return a validated result.

```
from fastapi import FastAPI, HTTPExceptionfrom pydantic import BaseModel, ConfigDict, Fieldfrom typing import Literalapp = FastAPI()class ReviewRequest(BaseModel):    model_config = ConfigDict(extra="forbid")    claim: str = Field(min_length=1, max_length=10_000)class Finding(BaseModel):    model_config = ConfigDict(extra="forbid")    claim: str    reason: str    policy_reference: str | Noneclass ReviewResponse(BaseModel):    model_config = ConfigDict(extra="forbid")    risk_level: Literal["low", "medium", "high"]    is_compliant: bool    findings: list[Finding]    requires_human_review: boolasync def run_model_review(claim: str) -> ReviewResponse:    """    Replace with your async model-service implementation.    It must enforce structured output and validate evidence.    """
```

Run the API after connecting the model service:

```
uvicorn main:app --reload
```

The `response_model=ReviewResponse` declaration documents the API contract and lets FastAPI validate the returned response. It does not force the underlying LLM to produce that schema. That responsibility belongs to the model-service layer.

For production, also add authentication, authorization, request limits, timeouts, tracing, and error handling. A valid schema must not allow a user to bypass access control or trigger an unauthorized compliance decision.

## 9. Structured output in multi-agent AI systems

In a LangGraph or multi-agent architecture, structured output is especially important because one agent's response becomes another component's input.

For example, in a compliance-review workflow:

Research agent

Returns retrieved policy evidence and source identifiers.

Compliance analysis agent

Returns typed findings, risk level, and evidence references.

Deterministic validator

Checks schema, evidence references, permissions, and business invariants.

Supervisor / workflow router

Routes to human approval, further research, or a final response based on validated state.

Each agent should have a clearly defined input/output contract. The supervisor should route using validated fields—not by guessing what an agent meant from its prose.

For example:

```
if review.requires_human_review:    next_step = "human_approval"elif not review.findings:    next_step = "complete"else:    next_step = "compliance_review"
```

This is a simplified routing example. In a real workflow, validate the state transition and ensure that the model cannot bypass required review merely by returning `requires_human_review=False`.

## 10. Final checklist: how do you reliably get JSON?

| Requirement                        | Recommended approach                             |
| ---------------------------------- | ------------------------------------------------ |
| Correct JSON syntax                | Native structured output or constrained decoding |
| Correct data types                 | Pydantic validation                              |
| Required fields and allowed values | Schema constraints                               |
| Valid policy citations             | Verify against the authoritative policy store    |
| Correct compliance decision        | Evidence checks, domain rules, and evaluation    |
| Temporary failures                 | Bounded retries with timeouts                    |
| Repeated invalid output            | Controlled failure or human escalation           |
| Reliable agent communication       | Typed contracts and validated state transitions  |

The most important distinction is:

- Prompting asks the model for JSON.
- Structured generation constrains the format.
- Pydantic validates the returned data.
- Business validation checks whether the data is actually acceptable.
- Workflow controls determine what happens next.

For a production financial AI platform, use all of these layers where appropriate. That is much more reliable than simply telling an LLM, “Return only JSON.”