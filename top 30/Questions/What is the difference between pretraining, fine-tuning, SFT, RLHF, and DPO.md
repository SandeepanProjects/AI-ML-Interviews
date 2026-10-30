# Pretraining vs Fine-Tuning vs SFT vs RLHF vs DPO

These five concepts are closely related, but they solve different problems in the development of an LLM.

As a Senior AI Engineer, you should understand not only their definitions but also:

- What happens to model weights?
- What data is required?
- What is the training objective?
- How does the model learn?
- When would you use each approach in production?

Let's understand them step by step.

# 1. First, understand the complete LLM training lifecycle

A typical LLM development pipeline looks like this:

Base Model Initialization

Randomly initialized model parameters

1\. Pretraining

Learn language and general patterns

Massive text and code datasets · Next-token prediction

2\. Fine-Tuning

Adapt the pretrained model to a specific domain or task. SFT is one common form.

3\. SFT (Supervised Fine-Tuning)

Learn to follow instructions

Curated prompt-response examples · Supervised learning

4\. Preference Alignment

Learn which responses are preferred by humans or preference criteria.

RLHF

Reward model + reinforcement learning

DPO

Direct preference optimization

Aligned Model → Evaluation → Deployment

Important: This is a common conceptual pipeline, not a mandatory sequence. Some models use different training approaches, and RLHF and DPO are alternative preference-optimization approaches in many pipelines.

# 2. Pretraining – Teaching the model general knowledge and language patterns

Stage 1

Pretraining is the initial large-scale training stage of a language model.

The objective is to teach the model how language and other training data are structured.

The model learns:

- Grammar.
- Language patterns.
- Relationships between words.
- Code patterns.
- Factual associations present in training data.
- Contextual representations.
- Some reasoning-like patterns.

## How does pretraining work?

Suppose we have this training sentence:

`The capital of India is New Delhi.`

The model receives tokenized input and learns to predict the next token.

For example:

| Input context               | Target token |
| --------------------------- | ------------ |
| The                         | capital      |
| The capital                 | of           |
| The capital of              | India        |
| The capital of India        | is           |
| The capital of India is     | New          |
| The capital of India is New | Delhi        |

In practice, training examples are processed in batches, and predictions at many token positions are calculated together using causal attention.

### Training architecture

[GPT-1: The Origin of Generative Pre-Training for Language Understanding - Interactive | Michael Brenndoerfer | Michael Brenndoerfer](https://images.openai.com/static-rsc-4/L5tIKEK9J7BRxlmWvUYzA9tzmT5eqZCzY3_nflwP7EDJPpupHga_0v6Btp7gyNnjvzoJmIn4FW40iXFvmSbsuykoguZDqnqFhxKBSb_O1-9-TntObMm1VrQb5CMiFctAUgL6C0VoSI1qw1geB2qfvlDHE2Yf9Q7VZuutmARfUJo?purpose=inline)

[mbrenndoerfer.com](https://mbrenndoerfer.com/writing/gpt-1-generative-pretraining-language-understanding?utm_source=chatgpt.com)

The training process:

1. Collect large-scale training data.
2. Clean and deduplicate the data.
3. Tokenize the data.
4. Feed token sequences into the Transformer.
5. Predict next-token probabilities.
6. Calculate cross-entropy loss.
7. Calculate gradients using backpropagation.
8. Update model weights using an optimizer.
9. Repeat over many batches.

### Training objective

A simplified next-token prediction objective:

\\[ L=-\sum\_{t=1}^{T}\log P\_\theta(x_t|x\_{\<t}) \\]

Where:

- \\(x_t\\) = actual token.
- \\(x\_{\<t}\\) = preceding tokens.
- \\(\theta\\) = model parameters.
- \\(P\_\theta\\) = model's predicted probability.

The model tries to maximize the probability of the correct next token.

### What happens to the weights?

The model starts with randomly initialized parameters.

During pretraining, millions or billions of parameter values are updated through gradient descent.

At the end, we obtain a pretrained base model.

Example: A pretrained base model may be able to complete text and understand many language patterns, but it may not reliably follow conversational instructions.

That is why further training is useful.

# 3. Fine-Tuning – Adapting an existing model to a specific requirement

Model adaptation

Fine-tuning means taking an already pretrained model and training it further using a more targeted dataset.

Instead of teaching the model general language patterns from scratch, we adapt its existing capabilities.

For example, suppose you have a general-purpose LLM.

You want to use it for:

- Financial analysis.
- Medical document classification.
- Legal document summarization.
- Customer support.
- SQL generation.
- Software development.

You can fine-tune the model using relevant examples.

## Example: Financial domain fine-tuning

Suppose you have 100,000 curated financial examples.

| Input                         | Expected output        |
| ----------------------------- | ---------------------- |
| Explain P/E ratio             | Financial explanation  |
| Analyze revenue growth        | Revenue analysis       |
| Explain market capitalization | Market-cap explanation |
| Identify financial risk       | Risk assessment        |

The pretrained model is further trained on this dataset.

[Customizing the Giants: Essential Techniques for Effective LLM Fine-Tuning | by Karan_bhutani | Medium](https://images.openai.com/static-rsc-4/OXCqZGj913lK8w6pJKSf6be9IAHjxNjX9n_R6pzBK-BS1MNg-MRdm48jq3PeArhFTWdIm0JgZts2MReF5MqkMqrKdEGg0T0pe3DZ0Z7ecO0MbwE00EKU6-BosxQp1LS-S9ZRv9ARnQfO9PF8ZDdkxSqhQAskcaMK0YgqxK4Y86g?purpose=inline)

[medium.com](https://medium.com/%40karanbhutani477/customizing-the-giants-essential-techniques-for-effective-llm-fine-tuning-298ff5e184a3?utm_source=chatgpt.com)

### What happens internally?

The process is:

```
Pretrained Model
       |
       v
Domain-specific Dataset
       |
       v
Forward Pass
       |
       v
Calculate Loss
       |
       v
Backpropagation
       |
       v
Update Model Parameters
       |
       v
Fine-tuned Model
```

The model retains useful capabilities learned during pretraining while adapting to the new data.

However, poor fine-tuning data or aggressive updates can cause catastrophic forgetting, where previously learned capabilities deteriorate.

## Types of fine-tuning

| Approach           | Description                                     |
| ------------------ | ----------------------------------------------- |
| Full fine-tuning   | Updates most or all model parameters            |
| LoRA               | Trains low-rank adapter matrices                |
| QLoRA              | Uses a quantized base model and trains adapters |
| SFT                | Supervised training on input-output examples    |
| Domain fine-tuning | Adapts behavior to a particular domain          |

Important distinction:

Fine-tuning is the broader concept. SFT is a particular training approach within fine-tuning.

Fine-tuning does not always mean SFT; it can also involve preference optimization and other objectives.

# 4. SFT (Supervised Fine-Tuning) – Teaching the model to follow instructions

Instruction tuning

SFT is one of the most commonly used fine-tuning techniques for conversational LLMs.

The objective is to teach the model how to respond to user instructions.

Remember:

- Pretraining teaches general language patterns.
- SFT teaches the model how to respond to instructions.

## How does SFT work?

Suppose you want to build a financial advisor assistant.

You create curated training examples containing prompts and expected responses.

Example:

SFT training example

Prompt

What is a mutual fund?

Expected response

A mutual fund is an investment vehicle that pools money from multiple investors and invests it in assets such as stocks, bonds and other securities.

The model receives the prompt and learns to generate the expected response.

### SFT architecture

[How to Train an LLM: 2026 Workflow Guide | Label Your Data](https://images.openai.com/static-rsc-4/uquupTkK3tyhJArVudybWQAeuH0OwMuGgSaUrxQDIUjPUCIVbfXeVTuNihYmBlib9WqU10vtorJYCOKNmZPlt2csqlkqwa1_WQCdjaa-5kK-EiKGOgJhUaKNVnP4SzSiH7I9GtPph3xZ0lrESD7POWtshUVYLDL4XyenVFGG1UE?purpose=inline)

[labelyourdata.com](https://labelyourdata.com/articles/how-to-train-an-llm?utm_source=chatgpt.com)

### Step-by-step process

1. Collect high-quality instruction-response examples.
2. Format them into the model's conversation template.
3. Tokenize the examples.
4. Feed the input and expected response into the model.
5. Calculate prediction loss on the target response tokens.
6. Backpropagate gradients.
7. Update model parameters.

An important implementation detail: SFT commonly uses a loss mask so that only the desired assistant-response tokens contribute to the loss, rather than training the model to reproduce the user's prompt.

### SFT loss function

SFT commonly uses the same autoregressive cross-entropy objective as pretraining, but on supervised examples.

\\[ L\_{SFT} = -\sum\_{t \in Response} \log P\_\theta(y_t|x,y\_{\<t}) \\]

Where:

- x = user prompt.
- y = expected response.
- θ = model parameters.

The model learns to predict the correct response tokens.

### Practical example

Suppose you are building an enterprise financial compliance assistant.

You want it to produce structured JSON.

Training example:

Input:

```
Analyze this financial statement
for regulatory compliance.
```

Expected output:

```
{
  "risk_level": "HIGH",
  "compliance_status": "REVIEW_REQUIRED",
  "reason": "Unsupported financial claim",
  "requires_human_review": true
}
```

You can use many high-quality examples of this type to teach the model the expected response format and task behavior.

However, SFT alone does not guarantee valid JSON, factual correctness or regulatory compliance. You still need output validation, retrieval grounding, guardrails and evaluation.

## When should you use SFT?

- You want consistent response formatting.
- You want better instruction following.
- You want domain-specific response behavior.
- You want to teach the model a particular task.
- You have a high-quality dataset of prompt-response pairs.

# 5. RLHF (Reinforcement Learning from Human Feedback)

Preference alignment

Now let's move to a more advanced concept.

SFT teaches the model to imitate good examples.

But how do we teach it to prefer a better answer when multiple answers are possible?

This is where RLHF comes in.

RLHF uses human preference feedback to improve model behavior.

## Why do we need RLHF?

Consider this question:

`Explain what RAG is.`

The model generates two possible responses.

User prompt

Explain what RAG is.

Response A

Preferred

RAG stands for Retrieval-Augmented Generation. It retrieves relevant information from an external knowledge source and provides that context to an LLM to generate a more grounded answer.

Response B\
RAG is a model that stores all documents inside its neural network and generates answers from its memory.

Response A is more accurate.

Human evaluators can compare these responses and indicate that A is better than B.

The model can then be trained to favor responses that receive better preference scores.

## RLHF has three major stages

[GPT Models & Text Generation - Complete NLP Series Part 10 - Wasil Zafar](https://images.openai.com/static-rsc-4/uiP5ahc60dAulEuDUvd-QqrPM__g5G9g9rixXlIJABeO-MgvNv_aN7Bu9eH4eOIMbLAvfLq5eo3TGBW4H4C2GKqBzyuYjY16wYyvzuVY0p7RY4pIXFApt8DlgRiRI3FrRfyosbXCU0CIvXzvO39r3FwjdYmQeJnhq0HZz_eQOBY?purpose=inline)

[wasilzafar.com](https://www.wasilzafar.com/pages/series/nlp/nlp-gpt-text-generation.html?utm_source=chatgpt.com)

### Stage 1: Supervised Fine-Tuning

First, train a pretrained model using high-quality instruction-response examples.

This creates an instruction-following policy model.

```
Pretrained Model
       |
       v
Instruction Dataset
       |
       v
Supervised Fine-Tuning
       |
       v
SFT Model
```

### Stage 2: Reward Model Training

Now collect human preference comparisons.

Example:

| Prompt      | Response   | Human preference |
| ----------- | ---------- | ---------------- |
| Explain RAG | Response A | Preferred        |
| Explain RAG | Response B | Rejected         |

A reward model is trained to score responses.

For example:

| Response | Reward score |
| -------- | ------------ |
| A        | 0.92         |
| B        | 0.21         |

These numbers are illustrative.

The reward model learns to predict which responses humans prefer.

A common reward-model objective is the Bradley-Terry preference loss:

\\[ L\_{RM} = -\log \sigma(r\_\phi(x,y_w)-r\_\phi(x,y_l)) \\]

Where:

- \\(r\_\phi\\) = reward model.
- \\(y_w\\) = preferred response.
- \\(y_l\\) = rejected response.
- \\(\sigma\\) = sigmoid function.

The reward model is not the same thing as the original language model.

It is a separate model that estimates the quality of a response according to learned preferences.

### Stage 3: Reinforcement Learning using PPO

Now we optimize the language model using the reward model.

A common traditional RLHF approach uses PPO (Proximal Policy Optimization).

The process:

```
User Prompt
     |
     v
Policy LLM
     |
     v
Generate Response
     |
     v
Reward Model
     |
     v
Calculate Reward
     |
     v
PPO Optimization
     |
     v
Updated Policy LLM
```

The policy model generates responses.

The reward model evaluates those responses.

PPO updates the policy to increase expected reward while constraining how much the policy changes.

A common objective includes a KL penalty against a reference model:

\\[ Objective = E[r\_\phi(x,y)] - \beta D\_{KL}(\pi\_\theta || \pi\_{ref}) \\]

Where:

- \\(r\_\phi\\) = learned reward.
- \\(\pi\_\theta\\) = current policy.
- \\(\pi\_{ref}\\) = reference policy.
- β = KL penalty coefficient.

The KL penalty discourages the updated model from moving too far away from the reference model.

In practical PPO-based RLHF, the actual optimization includes additional policy-gradient and clipping terms.

## Why is RLHF useful?

It helps improve:

1. Helpfulness.
2. Instruction following.
3. Response quality.
4. Preference alignment.
5. Certain safety-related behaviors.

However, reward models can be imperfect. A model can learn to exploit weaknesses in the reward model rather than genuinely improve response quality. This is called reward hacking or reward-model overoptimization.

# 6. DPO (Direct Preference Optimization)

Alternative to traditional RLHF

DPO is particularly important because it simplifies preference optimization.

The main idea:

Instead of training a separate reward model and then using PPO to optimize the LLM, DPO directly trains the language model using preferred and rejected response pairs.

Let's understand the difference.

Traditional RLHF:

```
Human Preferences
       |
       v
Train Reward Model
       |
       v
Generate Responses
       |
       v
PPO Optimization
       |
       v
Updated LLM
```

DPO:

```
Human Preferences
       |
       v
Preferred / Rejected Pairs
       |
       v
DPO Training
       |
       v
Updated LLM
```

[LLMs-from-scratch/ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb at main · rasbt/LLMs-from-scratch · GitHub](https://images.openai.com/static-rsc-4/MUPtUIcgUJ9OIa4JUG-5abQOBQeWoVjh7L5oNccZdMLpLrLn95IiINBakvsZaNm4qREfDxM4xI4qOXeu5pqlgKkJCLNH5MTX0-UBwNcBIXk8orXw2tZGdp8Y_RRbloMBz334OADpVWvqzL0SqMPI9w1Bql9MVc9xmdq1aEWgkl0?purpose=inline)

[github.com](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb?utm_source=chatgpt.com)

## 6.1 How does DPO work?

Suppose we have a prompt:

`Explain the importance of database indexing.`

Two responses are generated.

Preferred response:

Database indexing improves query performance by allowing the database to locate records without scanning the entire table.

Rejected response:

Database indexing is used to increase the size of a database.

We create a preference dataset:

```
{    "prompt": "Explain database indexing",    "chosen": "Database indexing improves query performance...",    "rejected": "Database indexing is used to increase..."}
```

DPO trains the model to increase the relative likelihood of the preferred response compared with the rejected response.

### 6.2 DPO mathematical objective

The DPO objective is based on a relationship between reward functions and an optimal policy relative to a reference policy.

A common DPO loss is:

\\[ L\_{DPO} = -\log \sigma \left( \beta \left[ \log\frac{\pi\_\theta(y_w|x)} {\pi\_{ref}(y_w|x)} - \log\frac{\pi\_\theta(y_l|x)} {\pi\_{ref}(y_l|x)} \right] \right) \\]

Where:

- \\(\pi\_\theta\\) = trainable language model.
- \\(\pi\_{ref}\\) = frozen reference model.
- \\(y_w\\) = preferred response.
- \\(y_l\\) = rejected response.
- β = controls the strength of the reference-policy constraint.
- σ = sigmoid function.

Let's simplify what this equation does.

It compares:

1. How likely the current model is to generate the preferred response.
2. How likely the current model is to generate the rejected response.
3. How those probabilities compare with the reference model.

The objective encourages the model to favor the preferred response relative to the rejected response, without moving excessively away from the reference policy.

Important: DPO does not mean there is no reward-related mathematics. It uses an implicit reward formulation, but does not require training a separate reward model or running a PPO optimization loop.

# 7. RLHF vs DPO – What is the actual difference?

This is one of the most frequently asked questions in senior AI interviews.

| Feature                    | RLHF (PPO-based)                  | DPO                                                     |
| -------------------------- | --------------------------------- | ------------------------------------------------------- |
| Human preference data      | Yes                               | Yes                                                     |
| Preferred/rejected pairs   | Commonly used                     | Required in standard DPO                                |
| Separate reward model      | Yes                               | No                                                      |
| PPO reinforcement learning | Yes, in traditional RLHF          | No                                                      |
| Reference model            | Commonly used                     | Used                                                    |
| Training complexity        | Higher                            | Generally lower                                         |
| Training infrastructure    | More components                   | Simpler pipeline                                        |
| Optimization               | Reward-driven policy optimization | Direct preference optimization                          |
| Reward hacking             | Possible                          | Other forms of preference overoptimization are possible |

### An important clarification

RLHF is a broad family of approaches, not a synonym for PPO.

Traditional PPO-based RLHF uses a reward model and reinforcement learning. Other approaches to learning from human feedback exist, including DPO.

DPO is not simply another name for RLHF; it is a distinct preference-optimization method.

# 8. Complete comparison of all five concepts

Let's put everything together.

| Parameter              | Pretraining               | Fine-tuning             | SFT                   | RLHF                       | DPO                        |
| ---------------------- | ------------------------- | ----------------------- | --------------------- | -------------------------- | -------------------------- |
| Primary purpose        | General language learning | Task/domain adaptation  | Instruction following | Preference alignment       | Preference alignment       |
| Training data          | Large text/code corpus    | Domain/task data        | Prompt-response pairs | Preference feedback        | Chosen/rejected pairs      |
| Learning objective     | Next-token prediction     | Depends on method       | Supervised prediction | Reward optimization        | Preference optimization    |
| Model weights updated? | Yes                       | Usually yes or adapters | Yes or adapters       | Yes                        | Yes                        |
| Separate reward model  | No                        | Not required            | No                    | In traditional RLHF, yes   | No                         |
| Complexity             | Very high                 | Variable                | Moderate              | High                       | Moderate                   |
| Typical use            | Base LLM creation         | Domain adaptation       | Chat assistants       | Human preference alignment | Human preference alignment |

# 9. Practical example: Building a Financial Advisor LLM

Let's say you are building an enterprise financial advisor copilot.

You want the model to:

- Understand financial concepts.
- Answer financial questions.
- Follow enterprise instructions.
- Produce structured JSON.
- Prefer accurate and responsible answers.

Here is how you might use these techniques.

## Financial Advisor LLM development

1. Pretraining

   Train a base model on large-scale text and code data.

   Result: General language and knowledge capabilities.
2. Domain fine-tuning

   Adapt the base model to financial terminology, financial analysis and relevant tasks.

   Result: Better specialization.
3. SFT

   Train on financial advisor instructions and structured response examples.

   Result: Better instruction following and consistent response formats.
4. RLHF or DPO

   Train using financial experts' preferences for accurate, clear and responsible responses.

   Result: Better preference alignment.
5. Evaluation and deployment

   Evaluate accuracy, hallucinations, safety, structured outputs and performance.

   Deploy through a serving engine such as vLLM.

   Result: Production-ready model, subject to passing the required evaluations.

One important architectural decision: You do not necessarily need to fine-tune an LLM to build this financial advisor.

For frequently changing financial data, use RAG to retrieve current information. Fine-tuning is more appropriate for teaching task behavior, response style and domain-specific patterns.

RAG and fine-tuning solve different problems and can be combined.

# 10. Practical implementation: How do you train using SFT and DPO?

Let's connect the concepts to actual Python code.

Two popular libraries are:

- Hugging Face Transformers.
- TRL.

## A. SFT training using Hugging Face TRL

Example using a pretrained model, LoRA adapters and supervised instruction data.

```
from datasets import Datasetfrom transformers import (    AutoModelForCausalLM,    AutoTokenizer,    TrainingArguments)from peft import LoraConfigfrom trl import SFTTrainer, SFTConfigmodel_name = "your-base-model"tokenizer = AutoTokenizer.from_pretrained(    model_name)model = AutoModelForCausalLM.from_pretrained(    model_name)if tokenizer.pad_token is None:    tokenizer.pad_token = tokenizer.eos_tokentrain_data = Dataset.from_list([    {        "text": (            "<|user|> What is a mutual fund?\n"            "<|assistant|> A mutual fund pools "            "investor money and invests it in assets."        )    },    {        "text": (            "<|user|> What is an index fund?\n"            "<|assistant|> An index fund tracks "            "a selected market index."        )
```

This is an illustrative starting point, not a production-ready training pipeline. The conversation template, assistant-only loss masking, EOS handling, dataset quality, validation split, hardware configuration and model-specific LoRA target modules must be checked before training.

## B. DPO training using Hugging Face TRL

Now imagine you have preference data.

```
from datasets import Datasetfrom transformers import (    AutoModelForCausalLM,    AutoTokenizer)from trl import DPOTrainer, DPOConfigmodel_name = "your-sft-model"model = AutoModelForCausalLM.from_pretrained(    model_name)tokenizer = AutoTokenizer.from_pretrained(    model_name)if tokenizer.pad_token is None:    tokenizer.pad_token = tokenizer.eos_tokenpreference_data = Dataset.from_list([    {        "prompt": "Explain database indexing.",        "chosen": (            "Database indexing improves query "            "performance by reducing full table scans."        ),        "rejected": (            "Database indexing increases the "            "size of a database."        )    }])training_args = DPOConfig(    output_dir="./financial-dpo",
```

With `ref_model=None`, current TRL configurations can use an implicit reference through the appropriate PEFT/reference handling or other version-specific behavior. Verify the exact reference-model behavior for your installed TRL version. For a rigorous experiment, explicitly control and evaluate the reference policy.

# 11. When should a Senior AI Engineer choose each technique?

This is where interviewers expect architectural decision-making, not just definitions.

| Requirement                                                             | Recommended approach    |
| ----------------------------------------------------------------------- | ----------------------- |
| Build a general-purpose base LLM                                        | Pretraining             |
| Adapt an existing model to financial tasks                              | Domain fine-tuning      |
| Teach consistent instruction following                                  | SFT                     |
| Optimize responses using human preference and PPO                       | RLHF                    |
| Optimize responses using chosen/rejected pairs with less infrastructure | DPO                     |
| Teach frequently changing company information                           | RAG, not fine-tuning    |
| Teach a specific output format                                          | SFT + schema validation |
| Improve preferred response behavior                                     | DPO or RLHF             |

## 12. Five important interview follow-up questions

Q1. Is SFT the same as fine-tuning?

No. Fine-tuning is the broader concept. SFT is one specific form of supervised fine-tuning using labeled input-output examples.

Q2. Can we directly apply DPO to a pretrained model?

Technically, yes, provided you have suitable preference data and a reference policy. However, a well-instructed SFT model is commonly used as the starting policy because it already understands conversational instructions.

Q3. Does DPO require a reward model?

No separate trained reward model is required. DPO directly optimizes the model using preference pairs and a reference policy.

Q4. Does RLHF always require PPO?

No. PPO is a traditional and widely discussed RLHF approach, but RLHF is a broader category of human-feedback-based training methods.

Q5. Should we fine-tune an LLM whenever it gives incorrect answers?

No. First identify the root cause.

- Missing knowledge or frequently changing information: RAG.
- Poor instruction following: SFT.
- Unwanted response behavior: DPO or RLHF.
- Incorrect reasoning or poor general capabilities: investigate model selection, training data, task design and evaluation.
- Invalid JSON: structured decoding and schema validation, potentially supported by SFT.

# 13. Test your understanding

Knowledge check

0/5 answered

1\. Which training stage primarily teaches general language patterns through next-token prediction?

SFT

Pretraining

DPO

RLHF

2\. Which approach uses prompt-response examples to teach instruction following?

SFT

DPO

PPO

KV caching

3\. Which traditional RLHF component predicts human preference scores?

Tokenizer

Embedding layer

Reward model

Vector database

4\. Which approach directly optimizes chosen and rejected responses without a separate reward model?

Pretraining

DPO

PPO-based RLHF

RAG

5\. Your financial LLM needs current stock prices. What should you use?

Pretraining

DPO

RAG or live data integration

SFT alone

Submit answers

My recommendation for your Senior AI Engineer preparation: Master these five concepts along with LoRA, QLoRA, PPO, reward modeling, gradient descent and catastrophic forgetting. Also understand when RAG is preferable to fine-tuning. Together, these topics help you explain the complete journey from a pretrained base model to an enterprise-ready aligned LLM.