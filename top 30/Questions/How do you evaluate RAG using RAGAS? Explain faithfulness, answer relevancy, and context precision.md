# How do you evaluate RAG using RAGAS?

This is an important Senior AI Engineer interview question because building a RAG pipeline is only half the job. You must also prove that your RAG system retrieves the correct information and generates answers grounded in the retrieved documents.

RAGAS (Retrieval Augmented Generation Assessment) is a framework for evaluating RAG applications using metrics that assess different stages of the RAG pipeline.

It helps answer three fundamental questions:

1. Is the LLM generating an answer supported by the retrieved context?
2. Is the generated answer actually relevant to the user's question?
3. Did the retriever retrieve the right information?

## 1. Understand the RAG pipeline first

[Demo: applying retrieval-augmented generation with open tools](https://images.openai.com/static-rsc-4/NgoA5-MkESKiEgRmOzvgDfAgO_K_z9DcEsWi-x8pLg6YlA5zRiJNNCHACs7By2b-MGcxcFxVL6K9QXPmAqThFKbUgtUl2nhZWMUoXAa0egaxTpQd0ufaCtLlfxSUIueIIka_chfpHt7T_GfYfpWaG_WaEDHPJTmsOfeXaFclVs8?purpose=inline)

[nebius.com](https://nebius.com/blog/posts/building-apps-on-top-of-llm/demo-rag-with-open-tools?utm_source=chatgpt.com)

A typical RAG pipeline works like this:

1. User submits a question.
2. The question is converted into an embedding.
3. The retriever searches the vector database.
4. Relevant document chunks are retrieved.
5. Retrieved chunks are passed to the LLM.
6. LLM generates the final answer.

The problem is that a RAG application can fail at different stages.

For example:

- Retrieval may return irrelevant documents.
- Retrieval may return relevant documents but miss important information.
- The LLM may hallucinate despite receiving correct context.
- The LLM may generate an answer that does not address the question.

RAGAS helps measure these problems separately.

## 2. The three important RAGAS metrics

1\. Faithfulness

Generation quality

Does the generated answer stay grounded in the retrieved context?

# Groundedness

2\. Answer relevancy

Generation quality

Does the generated answer actually address the user's question?

# Relevance

3\. Context precision

Retrieval quality

Are the retrieved chunks relevant, and are relevant chunks ranked near the top?

# Retrieval ranking

Let's understand each metric with practical examples.

## 3. Faithfulness

Definition: Faithfulness measures whether the generated answer is supported by the retrieved context.

It detects hallucinations introduced by the LLM.

Imagine the retrieved document contains:

```
Company ABC offers personal loans.

The maximum loan amount is ₹10 lakh.

The interest rate starts at 10% per annum.
```

User asks:

```
What is the maximum personal loan amount?
```

Faithful answer

Generated answer:

"The maximum personal loan amount offered by Company ABC is ₹10 lakh."

Unfaithful answer

Generated answer:

"Company ABC offers personal loans up to ₹25 lakh with an interest rate of 8%."

The second answer introduces information that is not supported by the retrieved document.

This is exactly what faithfulness is designed to detect.

How is faithfulness calculated?

RAGAS generally breaks the generated answer into individual claims or statements.

For example:

Generated answer:

```
Company ABC offers personal loans.
The maximum loan amount is ₹10 lakh.
The interest rate is 8%.
```

Claims:

1. Company ABC offers personal loans.
2. Maximum loan amount is ₹10 lakh.
3. Interest rate is 8%.

Suppose the retrieved context supports claims 1 and 2, but not claim 3.

\\[ Faithfulness = \frac{2}{3} = 0.667 \\]

Illustrative calculation; actual RAGAS evaluation uses an LLM-based claim verification process.

Senior engineer insight: Faithfulness does not mean that the answer is factually correct in the real world. It means that the answer is supported by the supplied context.

An outdated or incorrect source document can still produce a highly faithful answer.

## 4. Answer relevancy

Definition: Answer relevancy measures how well the generated answer addresses the original user question.

It checks whether the LLM answers what the user actually asked.

Important distinction:

- Faithfulness checks whether the answer is supported by the context.
- Answer relevancy checks whether the answer addresses the question.

Consider this example.

User asks:

```
What is the maximum personal loan amount offered by Company ABC?
```

Retrieved context:

```
Company ABC offers personal loans up to ₹10 lakh.

The interest rate starts at 10% per annum.

The repayment tenure is up to 5 years.
```

High answer relevancy

Generated answer:

"Company ABC offers personal loans up to ₹10 lakh."

Low answer relevancy

Generated answer:

"Company ABC was established in 1995. It offers several financial products, including personal loans, home loans and credit cards."

The second answer might contain correct information, but it doesn't answer the user's specific question.

How does RAGAS calculate answer relevancy?

The traditional RAGAS answer relevancy approach works roughly as follows:

1. Take the original user question.
2. Take the generated answer.
3. Use an evaluator LLM to generate possible questions that the answer could be responding to.
4. Convert those generated questions into embeddings.
5. Compare their semantic similarity with the original question.
6. Calculate the average similarity.

The conceptual formula is:

\\[ AR = \frac{1}{N}\sum\_{i=1}^{N} Similarity(q,q_i) \\]

Where:

- \\(q\\) = original user question.
- \\(q_i\\) = question generated from the answer.
- \\(N\\) = number of generated questions.

A score closer to 1 indicates better relevance.

This describes the traditional answer-relevancy metric; implementation details can vary between RAGAS versions.

Senior engineer insight: An answer can have high faithfulness but low answer relevancy.

For example, the LLM might accurately summarize three paragraphs from the retrieved document, even though the user only asked for one specific number.

That answer is grounded but unnecessarily broad.

## 5. Context precision

Definition: Context precision measures whether the retrieved document chunks are relevant to the user's question and whether relevant chunks appear higher in the retrieval ranking.

This evaluates the retriever rather than the generated answer.

Imagine a user asks:

```
What is the maximum personal loan amount?
```

Your vector database returns five chunks:

| Rank | Retrieved chunk                            | Relevant? |
| ---- | ------------------------------------------ | --------- |
| 1    | Company ABC home loan details              | No        |
| 2    | Company ABC personal loan maximum ₹10 lakh | Yes       |
| 3    | Company ABC credit card details            | No        |
| 4    | Company ABC personal loan interest rate    | Yes       |
| 5    | Company ABC savings account details        | No        |

Here, only two of the five retrieved chunks are relevant.

More importantly, the first relevant chunk appears at rank 2 rather than rank 1.

This means your retrieval ranking needs improvement.

How is context precision calculated?

RAGAS uses a ranking-aware precision calculation.

\\[ CP\@K = \frac{\sum\_{k=1}^{K} P\@k \times v_k} {\text{Number of relevant chunks in top K}} \\]

Where:

- \\(K\\) = number of retrieved chunks.
- \\(P\@k\\) = precision at rank \\(k\\).
- \\(v_k\\) = 1 if the chunk is relevant, otherwise 0.

For our example:

| Rank | Relevant? | Precision at rank |
| ---- | --------- | ----------------- |
| 1    | 0         | 0/1 = 0           |
| 2    | 1         | 1/2 = 0.5         |
| 3    | 0         | 1/3 = 0.33        |
| 4    | 1         | 2/4 = 0.5         |
| 5    | 0         | 2/5 = 0.4         |

Therefore:

\\[ CP\@5 = \frac{0+0.5+0+0.5+0}{2} \\]

\\[ \boxed{CP\@5=0.50} \\]

This is an illustrative ranking calculation.

How do you improve context precision?

As a Senior AI Engineer, I would investigate:

1. Embedding model quality.
2. Chunk size and overlap.
3. Metadata filtering.
4. Hybrid search using BM25 and vector search.
5. Cross-encoder reranking.
6. Top-K configuration.
7. Document duplication and outdated content.

For example, if Qdrant returns 20 chunks, I might introduce a reranking layer to select the best 5 chunks before passing them to the LLM.

[Hybrid Retrieval: Text + Vectors - Euron Daily | Euron Community](https://images.openai.com/static-rsc-4/VhpuM2YCGpQ7tmeJmZiShbcm2SfV3Z9YagFBkhIBjoEmUM7LVxoiRSKdChMCz5f_v6X82acmzr-pgd5y16JCLdPcGgYeFhuqL6aeSiSRNxc6_uFwoEbd0XA0B7bCzFczLqr-ndPS6NOmPra0jEkTrSCvR5nAcwie7ut0z73zlGg?purpose=inline)

[euron.one](https://euron.one/community/posts/71ff9edb-bbae-418a-bc00-cdd9393cd1df?utm_source=chatgpt.com)

## 6. Understand the difference between all three metrics

This is particularly important during interviews.

| Metric            | Evaluates | Main question                                      |
| ----------------- | --------- | -------------------------------------------------- |
| Faithfulness      | Generator | Is the answer supported by context?                |
| Answer relevancy  | Generator | Does the answer address the question?              |
| Context precision | Retriever | Are the retrieved chunks relevant and ranked well? |

There is another metric you should know: Context recall.

Context recall measures whether the retriever found all the information needed to answer the question.

For example, the document contains three requirements for loan eligibility, but the retriever only returns one.

- Context precision may be high because the returned chunk is relevant.
- Context recall may be low because two important pieces of information are missing.

This distinction is important when diagnosing retrieval failures.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://docs.ragas.io\&sz=32)

Ragas+2



# 7. Implement RAGAS evaluation in Python

Now let's look at how you would implement this in an enterprise RAG application.

## Step 1: Prepare your evaluation dataset

You need a representative dataset containing:

| Field                | Description                                          |
| -------------------- | ---------------------------------------------------- |
| `user_input`         | Original user question                               |
| `retrieved_contexts` | Chunks retrieved from Qdrant                         |
| `response`           | Answer generated by your LLM                         |
| `reference`          | Expected answer prepared by a human or domain expert |

Example:

```
evaluation_data = [    {        "user_input": "What is the maximum personal loan amount?",        "retrieved_contexts": [            "Company ABC offers personal loans up to ₹10 lakh.",            "The interest rate starts at 10% per annum.",            "Company ABC also provides home loans."        ],        "response": (            "Company ABC offers personal loans up to ₹10 lakh."        ),        "reference": (            "The maximum personal loan amount is ₹10 lakh."        )    }]
```

Where does this data come from?

In production, you should collect it from your actual RAG execution pipeline.

For every evaluation request, capture:

```
User Query
    |
    v
Retriever
    |
    v
Retrieved Chunks
    |
    v
LLM
    |
    v
Generated Response
    |
    v
Evaluation Dataset
```

The reference answer is generally maintained separately in a human-reviewed golden dataset.

## Step 2: Install dependencies

RAGAS APIs have evolved, so use the collections-based API for newer versions rather than copying older `single_turn_ascore()` examples.

```
pip install ragas openai
```

Set your API key securely:

```
export OPENAI_API_KEY="your-api-key"
```

## Step 3: Configure the evaluation models

RAGAS uses an evaluator LLM to judge claims, relevance, and retrieved contexts.

The evaluator should ideally be different from the model generating your production answer, so you can independently assess it.

For example:

```
from openai import AsyncOpenAIfrom ragas.llms import llm_factoryclient = AsyncOpenAI()evaluator_llm = llm_factory(    "gpt-4o-mini",    client=client)
```

For answer relevancy, configure an embedding model as well.

## Step 4: Implement the three metrics

The following uses the newer collections-based metric interface.

```
from ragas.metrics.collections import (    Faithfulness,    AnswerRelevancy,    ContextPrecision)from ragas.embeddings import OpenAIEmbeddingsembeddings = OpenAIEmbeddings(    model="text-embedding-3-small")faithfulness_metric = Faithfulness(    llm=evaluator_llm)answer_relevancy_metric = AnswerRelevancy(    llm=evaluator_llm,    embeddings=embeddings)context_precision_metric = ContextPrecision(    llm=evaluator_llm)
```

For the reference-based context precision metric, supply the expected answer when evaluating each sample.

## Step 5: Execute evaluation

```
import asyncioasync def evaluate_rag_sample(sample):    faithfulness = await faithfulness_metric.ascore(        user_input=sample["user_input"],        response=sample["response"],        retrieved_contexts=sample["retrieved_contexts"]    )    answer_relevancy = await answer_relevancy_metric.ascore(        user_input=sample["user_input"],        response=sample["response"]    )    context_precision = await context_precision_metric.ascore(        user_input=sample["user_input"],        response=sample["response"],        retrieved_contexts=sample["retrieved_contexts"],        reference=sample["reference"]    )    return {        "faithfulness": faithfulness.value,        "answer_relevancy": answer_relevancy.value,        "context_precision": context_precision.value    }async def main():    for sample in evaluation_data:        scores = await evaluate_rag_sample(sample)
```

API note: This illustrates the RAGAS 0.4+ collections-style scoring pattern. Verify the installed RAGAS version and embedding-provider interface when pinning dependencies. Run evaluation asynchronously with bounded concurrency for larger datasets.

The three metrics return scores between 0 and 1. RAGAS also supports returning evaluation explanations, depending on the metric and version.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://docs.ragas.io\&sz=32)

Ragas+2



The embedding initialization above should include the OpenAI client. Use:

```
from ragas.embeddings import OpenAIEmbeddingsembeddings = OpenAIEmbeddings(    client=client,    model="text-embedding-3-small")
```

The current RAGAS documentation describes this embedding interface.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://docs.ragas.io\&sz=32)

Ragas+1



# 8. How to interpret RAGAS scores in production

Suppose your evaluation produces the following results:

## RAG evaluation results

Faithfulness

# 0.65

Answer grounding

Answer relevancy

# 0.92

Question relevance

Context precision

# 0.45

Retrieval ranking

How would you investigate?

| Metric            | Score | Interpretation                           |
| ----------------- | ----- | ---------------------------------------- |
| Faithfulness      | 0.65  | Some answers contain unsupported claims  |
| Answer relevancy  | 0.92  | Answers generally address questions      |
| Context precision | 0.45  | Too many irrelevant chunks are retrieved |

My investigation would be:

1. First, inspect the retrieved chunks. The low context precision indicates a retrieval ranking problem.
2. Investigate embedding quality and chunking.
3. Introduce hybrid search and cross-encoder reranking.
4. Investigate hallucinations separately by examining the generated claims.
5. Improve the system prompt to require answers grounded in retrieved context.
6. Rerun the same golden evaluation dataset.

Important: These are illustrative scores, not universal production acceptance thresholds. Establish thresholds using your own domain, human-reviewed data, and risk tolerance.

# 9. How I would use RAGAS as a Staff AI Engineer

I would not run RAGAS only once after building the application.

I would integrate evaluation into the complete AI development lifecycle.

[The LLM Evaluation Playbook Every AI Engineer Needs | by Suresh Kumar Ariya Gowder | Think in AI Agents | Apr, 2026 | Medium](https://images.openai.com/static-rsc-4/AFVyw_lLAwGT3HiDYhV0rfJP7LM9TVKr4M7883QXfCt8qZd5-LyJJ7ii74lPCGGMcCy0TgXdRMl_11FO7sFaZv-u-gUHE6JksQuxgHr8cHbI7k7ong_6wIIfogXpFHwXsdwnYDe-zl76lApY0EOksbWNlmLaHyuvCgeHJa65MZU?purpose=inline)

[medium.com](https://medium.com/system-design-mastery-series/the-llm-evaluation-playbook-every-ai-engineer-needs-ae31eb603c83?utm_source=chatgpt.com)

| Stage          | What I would do                                        |
| -------------- | ------------------------------------------------------ |
| Development    | Evaluate retrieval and generation independently        |
| Pull request   | Run regression tests on golden questions               |
| Pre-production | Compare new model and prompt versions                  |
| Deployment     | Run canary evaluation                                  |
| Production     | Sample real queries for quality monitoring             |
| Incident       | Investigate metric degradation and identify root cause |

I would also monitor:

- Average faithfulness.
- Answer relevancy.
- Context precision and recall.
- P95 response latency.
- Token consumption.
- Cost per request.
- Retrieval failures.
- No-answer rate.

For a financial application, I would additionally create domain-specific evaluations for numerical accuracy, policy compliance, source citations, and financial claim verification.

## 10. Common interview follow-up questions

These are questions an interviewer may ask after you explain RAGAS.

Q1. Can faithfulness be high while answer correctness is low?

Yes. Suppose a document contains outdated financial information. The LLM faithfully reproduces that information, but the information is factually outdated.

Q2. Can answer relevancy be high while faithfulness is low?

Yes. The LLM may answer exactly what the user asked but invent the answer.

Q3. Can context precision be high while context recall is low?

Yes. The retriever might return one highly relevant chunk but miss other important chunks.

Q4. Does RAGAS require human-annotated ground truth for every metric?

No. Faithfulness and answer relevancy can be evaluated without a reference answer. Reference-based context precision and context recall require reference information. RAGAS also offers variants that operate without references.

Q5. Is RAGAS completely reliable?

No. It uses LLM-based judges for many metrics. Evaluator bias, model limitations, prompt sensitivity, and domain-specific reasoning can affect scores. I would calibrate it against human evaluations.

# 11. Interview-ready answer (what you should say)

If the interviewer asks, "How do you evaluate RAG using RAGAS?", you can answer:

> I use RAGAS to evaluate both retrieval and generation quality in my RAG architecture.
>
> I primarily focus on three metrics: faithfulness, answer relevancy, and context precision.
>
> Faithfulness measures whether the generated answer is supported by the retrieved context and helps identify hallucinations.
>
> Answer relevancy measures whether the generated answer actually addresses the user's question.
>
> Context precision measures whether the retrieved chunks are relevant and properly ranked.
>
> I also monitor context recall to ensure that the retriever is not missing important information.
>
> I maintain a golden evaluation dataset containing user questions, retrieved contexts, generated answers, and reference answers.
>
> I integrate RAGAS into our CI/CD pipeline to detect quality regressions whenever we change the embedding model, chunking strategy, retrieval configuration, prompt, or LLM.
>
> In production, I combine these metrics with OpenTelemetry traces, latency monitoring, token consumption, and human feedback to identify whether a problem originates in retrieval or generation.

My recommendation: Master the distinction between faithfulness, answer relevancy, context precision, and context recall. Interviewers often give you a failing RAG scenario and expect you to identify which component needs fixing rather than simply defining the metrics.

Official reference: [RAGAS metrics documentation](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/?utm_source=chatgpt.com).