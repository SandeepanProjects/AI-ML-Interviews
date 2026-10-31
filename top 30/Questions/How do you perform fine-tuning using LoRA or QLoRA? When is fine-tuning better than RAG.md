# Fine-tuning with LoRA and QLoRA — and when to choose it over RAG

At a Senior/Staff AI Engineer level, the most important distinction is:

- RAG changes what the model knows at inference time by retrieving external information and putting it into the context.
- Fine-tuning changes how the model behaves by training its parameters on examples.
- LoRA and QLoRA are parameter-efficient fine-tuning techniques that adapt a pretrained model without updating all its parameters.

For your Enterprise Financial Advisor Copilot, I would generally use RAG for changing financial policies, market information, and customer-specific portfolio data, while considering LoRA/QLoRA for consistent financial reasoning formats, classification, domain-specific language, and structured responses.

Let's walk through the architecture, training process, code, evaluation, and decision criteria.

## 1. What is fine-tuning?

Suppose you have a pretrained LLM:

```
Input:
Explain the risk associated with investing in a volatile asset.

Pretrained LLM:
Provides a general explanation.
```

You want it to consistently produce answers in your financial application's format:

```
{
  "risk_category": "HIGH",
  "risk_factors": [
    "High volatility",
    "Market uncertainty",
    "Liquidity risk"
  ],
  "recommendation": "Review portfolio allocation",
  "requires_human_review": true
}
```

You could repeatedly write detailed prompts to achieve this behavior. But if you need this behavior consistently across thousands of examples, fine-tuning may be useful.

Fine-tuning trains the model on examples of desired input-output behavior.

### Fine-tuning pipeline

Pretrained LLM

Llama, Qwen, Mistral, etc.

Training dataset

Input prompts + high-quality expected outputs

LoRA / QLoRA training

Train a small set of adapter parameters

Fine-tuned model

Base model + trained adapter

Inference application

FastAPI + LangGraph + model serving

The model learns patterns from your examples. It does not automatically become a reliable database of your latest financial policies.

## 2. What is LoRA?

LoRA stands for Low-Rank Adaptation.

In traditional fine-tuning, we update a large number of model parameters. For a model with billions of parameters, this can require substantial GPU memory and compute.

LoRA reduces the training cost by freezing the original model weights and training small additional matrices.

### Traditional fine-tuning

Suppose a pretrained model contains a weight matrix:

\\[ W \in \mathbb{R}^{d \times k} \\]

Traditional fine-tuning updates the matrix:

\\[ W' = W + \Delta W \\]

Here, the model learns changes to the original weight matrix.

### LoRA fine-tuning

LoRA approximates the weight update using two smaller matrices:

\\[ \Delta W = BA \\]

Where:

- \\(A\\) is a low-rank matrix.
- \\(B\\) is another low-rank matrix.
- \\(r\\) is the rank, generally much smaller than the original dimensions.

The resulting weight is:

\\[ W' = W + BA \\]

The original \\(W\\) remains frozen. Only \\(A\\) and \\(B\\) are trained.

[LoRA Fine-Tuning: Low-Rank Adaptation of Large Neural Networks - FeynmanWiki](https://images.openai.com/static-rsc-4/BKlrk5v3Wcb49rXtVKYL3yqXKKAi5qbhJI51bQOo3_IAkyxh4UVbRLCW8Cs-CenIG-d68WUffb_kmIdU2QjpfChJ5sseqtzZ9ZkWyLr1YFxTcP5tWlQqjXRKRFou2sHrvXCGr9mvOECgb59aRsxQIIMkf7ucn78ouzlFlm8OqY0?purpose=inline)

[feynmanwiki.com](https://www.feynmanwiki.com/library/lora-fine-tuning-low-rank-adaption-9zjy?utm_source=chatgpt.com)

### Example

Assume:

- Original matrix: 4096 × 4096
- LoRA rank: 8

Original parameter count:

\\[ 4096 \times 4096 = 16,777,216 \\]

LoRA parameters:

\\[ 4096 \times 8 + 8 \times 4096 = 65,536 \\]

That is approximately 256 times fewer trainable parameters for this matrix.

This is why LoRA is significantly more memory-efficient than full fine-tuning.

Important: LoRA does not necessarily train only one layer. You can apply adapters to multiple attention and, optionally, MLP projection layers.

## 3. What is QLoRA?

QLoRA stands for Quantized Low-Rank Adaptation.

It combines two techniques:

1. Quantization: load the pretrained model using low-bit weights, commonly 4-bit.
2. LoRA: train small adapter matrices while keeping the quantized base model frozen.

[Memory-efficient Fine-tuning with with QLoRA | Niklas Heidloff](https://images.openai.com/static-rsc-4/-j4hVuNQ_QtJCPEVZy73jBhNxtjFyRtaYN1NGGuq6XsI2p9gdBnT4vuL94IFLlq_E6DNUXFzuLr6HtXO9Zc6mW1LJjAN0_KYiNCHgQ1EjWjl5pHRJmZ_tLltcNmcJD1Kc7jvY8l2ZF6bpOE-l-87sd_PnVkWEzzfb9wfqZVs888?purpose=inline)

[heidloff.net](https://heidloff.net/article/qlora/?utm_source=chatgpt.com)

### LoRA vs QLoRA

| Feature             | LoRA                    | QLoRA                           |
| ------------------- | ----------------------- | ------------------------------- |
| Base model          | Usually FP16/BF16       | Usually 4-bit                   |
| Base weights        | Frozen                  | Frozen and quantized            |
| Adapter weights     | Trainable               | Trainable                       |
| GPU memory          | Lower than full FT      | Usually lower than LoRA         |
| Training complexity | Relatively simple       | More quantization configuration |
| Best use            | When GPU memory permits | When GPU memory is constrained  |

QLoRA commonly uses:

- NF4 (NormalFloat4) quantization
- Double quantization
- Paged optimizers
- BF16 or FP16 computation, depending on hardware

Why NF4?

Neural network weights often have an approximately normal distribution. NF4 is designed to represent normally distributed weights efficiently with 4-bit values.

Why double quantization?

It further compresses the quantization constants, reducing memory overhead.

Why paged optimizers?

They help manage optimizer memory spikes, including by using unified memory mechanisms.

Remember: QLoRA does not mean the entire training process runs in 4-bit. The base weights are quantized, while adapter computation and optimizer states use their appropriate higher-precision representations.

## 4. When should you use LoRA versus QLoRA?

Imagine you want to fine-tune a 7-billion-parameter instruction model.

Use LoRA when:

- You have enough GPU memory to load the base model in BF16/FP16.
- You want relatively straightforward adapter training.
- You have a GPU with sufficient memory for the model, activations, and optimizer states.

Use QLoRA when:

- Your GPU memory is limited.
- You want to fine-tune a larger model on a smaller GPU.
- You can accept the extra quantization configuration and potential quality/performance tradeoffs.

The actual GPU requirement depends on sequence length, batch size, gradient accumulation, model architecture, and checkpointing—not just the number of parameters.

## 5. End-to-end implementation using QLoRA

Let's build a small financial-risk classification model.

Objective: Given a financial scenario, return a structured risk assessment.

We will use:

- Python
- Hugging Face Transformers
- PEFT
- TRL / supervised fine-tuning concepts
- BitsAndBytes
- PyTorch

### Project structure

```
financial-llm-finetuning/
│
├── data/
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
│
├── training/
│   ├── prepare_dataset.py
│   ├── train.py
│   └── config.py
│
├── inference/
│   └── predict.py
│
├── evaluation/
│   └── evaluate.py
│
└── requirements.txt
```

### Step 1: Prepare your training dataset

Fine-tuning quality depends heavily on the quality and consistency of your examples.

Example `data/train.jsonl`:

```
{"instruction":"Assess the risk of this investment scenario.","input":"A customer invests 70% of their portfolio in a highly volatile cryptocurrency.","output":"{\"risk_category\":\"HIGH\",\"risk_factors\":[\"High asset concentration\",\"High volatility\",\"Potential liquidity risk\"],\"recommendation\":\"Review allocation and diversification\",\"requires_human_review\":true}"}
{"instruction":"Assess the risk of this investment scenario.","input":"A customer invests in a diversified portfolio of government bonds with a long-term investment horizon.","output":"{\"risk_category\":\"LOW_TO_MODERATE\",\"risk_factors\":[\"Interest rate risk\",\"Inflation risk\"],\"recommendation\":\"Review duration and investment objectives\",\"requires_human_review\":false}"}
{"instruction":"Assess the risk of this investment scenario.","input":"A customer invests most of their retirement savings in one small-cap stock.","output":"{\"risk_category\":\"HIGH\",\"risk_factors\":[\"Concentration risk\",\"Equity volatility\",\"Potential liquidity risk\"],\"recommendation\":\"Evaluate diversification and suitability\",\"requires_human_review\":true}"}
```

These are illustrative training examples, not validated financial advice or an authoritative risk policy.

Your real dataset should contain:

- Thousands of representative examples, if available.
- Correct and consistent expected responses.
- Normal and edge-case scenarios.
- Examples of uncertainty and refusal.
- Examples where the correct response is to request additional information.
- Versioned labels and domain-expert review.

Do not randomly split near-duplicate examples across train and test. That creates data leakage and makes evaluation look better than it really is.

### Step 2: Install dependencies

Example `requirements.txt`:

```
torch
transformers
datasets
peft
bitsandbytes
accelerate
```

For a reproducible deployment, pin and test compatible package versions and CUDA dependencies together.

### Step 3: Load the dataset and prepare training examples

We need to convert each example into a chat conversation.

The model should learn the desired assistant response—not simply memorize the entire input and output as an undifferentiated text sequence.

`training/prepare_dataset.py`

```
from datasets import load_datasetfrom transformers import AutoTokenizerMODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"MAX_LENGTH = 1024tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)if tokenizer.pad_token is None:    tokenizer.pad_token = tokenizer.eos_tokendef build_messages(example):    return [        {            "role": "system",            "content": (                "You are a financial risk analysis assistant. "                "Return valid JSON matching the requested risk schema. "                "Do not invent missing financial information."            ),        },        {            "role": "user",            "content": (                f"{example['instruction']}\n\n"                f"Scenario: {example['input']}"            ),        },    ]def tokenize_example(example):    messages = build_messages(example)    # Prompt ends with the assistant generation marker.
```

Why do we use `-100` in labels?

In PyTorch's causal language model loss, `-100` is ignored. We mask the prompt so the model is trained to predict the assistant completion rather than being rewarded for reproducing the prompt.

One implementation detail: prompt and full-conversation tokenization must align correctly for the selected tokenizer's chat template. Test several decoded examples and verify that the first unmasked label is at the beginning of the assistant answer. Truncation must not remove the entire answer.

### Step 4: Configure QLoRA

This is the most important part.

`training/config.py`

```
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"OUTPUT_DIR = "artifacts/financial-risk-qlora"MAX_LENGTH = 1024LORA_R = 16LORA_ALPHA = 32LORA_DROPOUT = 0.05LEARNING_RATE = 2e-4BATCH_SIZE = 1GRADIENT_ACCUMULATION_STEPS = 8EPOCHS = 3
```

What do these parameters mean?

| Parameter                     | Purpose                                    |
| ----------------------------- | ------------------------------------------ |
| `r`                           | LoRA rank; controls adapter capacity       |
| `lora_alpha`                  | Scales the LoRA update                     |
| `lora_dropout`                | Regularizes adapter training               |
| `learning_rate`               | Controls optimizer step size               |
| `batch_size`                  | Examples processed per device per step     |
| `gradient_accumulation_steps` | Simulates a larger batch                   |
| `epochs`                      | Number of passes through the training data |

For example, with one GPU:

\\[ \text{Effective batch size} = 1 \times 8 \times 1 = 8 \\]

For multiple GPUs, multiply by the number of devices, assuming the same per-device settings.

### Step 5: Train using QLoRA

`training/train.py`

```
import torchfrom datasets import load_from_diskfrom transformers import (    AutoModelForCausalLM,    AutoTokenizer,    BitsAndBytesConfig,    DataCollatorForSeq2Seq,    Trainer,    TrainingArguments,)from peft import (    LoraConfig,    get_peft_model,    prepare_model_for_kbit_training,)from config import (    MODEL_NAME,    OUTPUT_DIR,    LORA_R,    LORA_ALPHA,    LORA_DROPOUT,    LEARNING_RATE,    BATCH_SIZE,    GRADIENT_ACCUMULATION_STEPS,    EPOCHS,)def load_tokenizer():    tokenizer = AutoTokenizer.from_pretrained(        MODEL_NAME,        use_fast=True,    )
```

A few important production notes about this example:

1. It assumes a compatible CUDA environment and a single-process/single-GPU training setup. For distributed training, configure Accelerate/DeepSpeed and device placement appropriately; `device_map="auto"` should not be blindly reused for distributed training.
2. Check that all specified `target_modules` actually exist in your chosen model. Different architectures use different names.
3. The example uses `paged_adamw_8bit`, which requires compatible bitsandbytes support.
4. Verify that your installed Transformers version supports the `eval_strategy` argument shown here.
5. Monitor GPU memory, gradient norms, training loss, validation loss, and checkpoints. Do not select a model based only on the lowest training loss.

This is a practical single-GPU training template. Production use still requires dataset validation, distributed-training configuration where needed, checkpoint recovery, experiment tracking, and evaluation gates.

## 6. How would you perform LoRA instead of QLoRA?

The training process is almost identical. The main difference is how you load the pretrained model.

For LoRA, load the base model in BF16 or FP16 instead of 4-bit quantization.

Replace `load_qlora_model()` with:

```
def load_lora_model():    compute_dtype = (        torch.bfloat16        if torch.cuda.is_available()        and torch.cuda.is_bf16_supported()        else torch.float16    )    model = AutoModelForCausalLM.from_pretrained(        MODEL_NAME,        torch_dtype=compute_dtype,        device_map="auto",    )    model.config.use_cache = False    return model
```

Then attach the same `LoraConfig` using `get_peft_model()`.

| Training operation  | LoRA      | QLoRA     |
| ------------------- | --------- | --------- |
| Load base model     | BF16/FP16 | 4-bit NF4 |
| Prepare k-bit model | No        | Yes       |
| Freeze base weights | Yes       | Yes       |
| Train adapters      | Yes       | Yes       |
| Save adapter        | Yes       | Yes       |

The rest of the training pipeline—dataset preparation, loss masking, optimizer, evaluation, and adapter saving—can remain largely the same.

## 7. How do you use the fine-tuned model for inference?

After training, you generally have two artifacts:

- Original pretrained model.
- Fine-tuned LoRA adapter weights.

You load the base model and attach the trained adapter.

`inference/predict.py`

```
import torchfrom transformers import (    AutoModelForCausalLM,    AutoTokenizer,    BitsAndBytesConfig,)from peft import PeftModelBASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"ADAPTER_PATH = "artifacts/financial-risk-qlora"def load_model():    compute_dtype = (        torch.bfloat16        if torch.cuda.is_available()        and torch.cuda.is_bf16_supported()        else torch.float16    )    tokenizer = AutoTokenizer.from_pretrained(        ADAPTER_PATH    )    quant_config = BitsAndBytesConfig(        load_in_4bit=True,        bnb_4bit_quant_type="nf4",        bnb_4bit_use_double_quant=True,        bnb_4bit_compute_dtype=compute_dtype,    )
```

For a real serving application, load the model once during application startup—not once per API request. Also, validate the generated JSON with a strict Pydantic schema before returning it.

For example:

```
from typing import Literalfrom pydantic import BaseModel, ConfigDictclass RiskAssessment(BaseModel):    model_config = ConfigDict(extra="forbid")    risk_category: Literal[        "LOW",        "LOW_TO_MODERATE",        "MODERATE",        "HIGH",    ]    risk_factors: list[str]    recommendation: str    requires_human_review: bool
```

The schema is a validation boundary, not a guarantee that the risk assessment is factually correct. You still need business-rule validation and, for regulated decisions, appropriate human review.

## 8. How do you evaluate a fine-tuned model?

This is an important Staff Engineer interview topic.

A model that has a lower training loss is not necessarily a better production model.

I would evaluate it at five levels:

| Evaluation     | What to measure                                      |
| -------------- | ---------------------------------------------------- |
| Training       | Training loss, validation loss, overfitting          |
| Task quality   | Accuracy, precision, recall, F1                      |
| Output quality | JSON validity, required fields, semantic correctness |
| Safety         | Hallucination, unsafe advice, refusal behavior       |
| Production     | Latency, throughput, GPU memory, cost                |

For our financial risk classification model:

- Does it correctly identify high-risk scenarios?
- Does it identify concentration risk?
- Does it invent missing portfolio information?
- Does it follow the expected JSON schema?
- Does it appropriately request human review?
- Does it generalize to scenarios not seen during training?

### Example evaluation code

```
import jsonfrom pydantic import ValidationErrorfrom inference.predict import predictfrom schemas import RiskAssessmentdef evaluate_example(example):    prediction = predict(example["scenario"])    try:        parsed = RiskAssessment.model_validate_json(            prediction        )        valid_json = True    except ValidationError:        parsed = None        valid_json = False    return {        "valid_json": valid_json,        "predicted_category": (            parsed.risk_category if parsed else None        ),        "expected_category": example["expected_category"],        "category_correct": (            parsed is not None            and parsed.risk_category            == example["expected_category"]        ),    }
```

In a real evaluation harness, I would add:

- A versioned held-out test set.
- Confusion matrix and per-class precision/recall.
- Domain-expert review of recommendations.
- Adversarial and out-of-distribution examples.
- Comparisons against the original base model.
- Comparisons against a prompt-only baseline.
- Confidence and abstention analysis.
- Repeated evaluation across model and adapter versions.

For regulated financial workflows, do not optimize only for accuracy. A false LOW-risk classification may be much more consequential than a false HIGH-risk classification. Use class-specific error costs and explicit business thresholds.

## 9. When is fine-tuning better than RAG?

This is the most important conceptual question.

Consider two requirements.

Requirement A: The model must know the latest financial compliance policy.

Requirement B: The model must consistently classify financial risk using your organization's preferred classification format.

These are different problems.

For Requirement A, use RAG.

For Requirement B, consider fine-tuning.

### RAG versus fine-tuning

RAG

Best for knowledge that changes frequently

[Demo: applying retrieval-augmented generation with open tools](https://images.openai.com/static-rsc-4/NgoA5-MkESKiEgRmOzvgDfAgO_K_z9DcEsWi-x8pLg6YlA5zRiJNNCHACs7By2b-MGcxcFxVL6K9QXPmAqThFKbUgtUl2nhZWMUoXAa0egaxTpQd0ufaCtLlfxSUIueIIka_chfpHt7T_GfYfpWaG_WaEDHPJTmsOfeXaFclVs8?purpose=inline)

[nebius.com](https://nebius.com/blog/posts/building-apps-on-top-of-llm/demo-rag-with-open-tools?utm_source=chatgpt.com)

1. User asks a question.
2. Retrieve relevant documents.
3. Add retrieved content to the model context.
4. Generate a grounded response.

Examples: current financial policies, customer portfolios, market reports, product documentation.

Fine-tuning

Best for learning consistent behavior

[How to run performance and scale validation for OpenShift AI | Red Hat Developer](https://images.openai.com/static-rsc-4/1xCmhN_4ALlXw8f_2lmnGHPraGpiPPk_a0D1JslLG6v_KiWXFLwpNcdpQrj-qrmrMTA1RmESaNaYutHppNsD4B7LPZcYJXicAZlfy03JhKMVOxczw4UV1tOiRtUK_CZilvb21MDEsKHZFMPNk-oz18mBOCXXPl3sK-Myu54pNxo?purpose=inline)

[developers.redhat.com](https://developers.redhat.com/articles/2025/04/30/how-run-performance-and-scale-validation-openshift-ai?utm_source=chatgpt.com)

1. Prepare high-quality examples.
2. Train model adapters.
3. Validate the resulting model.
4. Serve the adapted model.

Examples: classification, response style, domain terminology, structured output patterns, task-specific behavior.

### Detailed comparison

| Dimension             | RAG                              | Fine-tuning                                      |
| --------------------- | -------------------------------- | ------------------------------------------------ |
| Main purpose          | Provide external knowledge       | Adapt model behavior                             |
| Knowledge updates     | Update documents/index           | Usually requires retraining for reliable updates |
| Citations             | Can cite retrieved sources       | Does not inherently provide citations            |
| Private documents     | Retrieve with access controls    | Training can expose memorized information        |
| Domain-specific style | Prompting or examples            | Can learn consistent patterns                    |
| Structured output     | Prompt/schema constrained        | Can improve consistency, but still validate      |
| Upfront work          | Ingestion and retrieval          | Dataset preparation and training                 |
| Inference overhead    | Retrieval and context processing | No retrieval required unless combined with RAG   |
| Best for              | Dynamic facts                    | Repeated task patterns                           |

### Cases where fine-tuning is better

1\. Consistent classification

Suppose you have 50,000 reviewed examples of financial transaction classification.

You want the model to categorize each transaction into:

- Equity
- Fixed income
- Alternative investment
- Cash equivalent
- Other

Fine-tuning can teach the model this task consistently.

2\. Domain-specific output format

Suppose every response must follow an organization's standard structure, including specific fields, terminology, and explanation style.

Fine-tuning can improve consistency, although constrained decoding and schema validation are still important.

3\. Specialized task behavior

Examples:

- Extracting financial entities from documents.
- Classifying compliance violations.
- Categorizing customer support requests.
- Identifying document types.
- Producing standardized financial summaries.

4\. Repeated behavior that is expensive to prompt

If every request requires a large set of repeated instructions and examples, a fine-tuned model may reduce prompt size and improve consistency. You must measure the total cost, including training, hosting, and maintenance.

### Cases where RAG is better

Use RAG when:

- Policies change frequently.
- Documents are added and deleted regularly.
- Answers need source citations.
- Customer-specific data must be retrieved securely.
- Different users have different document permissions.
- You need traceability to the source of an answer.

For example, if a financial policy changes tomorrow, updating your RAG document index is generally more appropriate than retraining a model.

## 10. Can we combine RAG with fine-tuning?

Absolutely. In enterprise AI applications, this is often the best architecture.

For your Financial Advisor Copilot, I would consider the following design:

User query

Assess portfolio risk and explain applicable policy requirements

FastAPI + authentication + tenant authorization

Validate user, tenant, permissions and request

RAG pipeline

PostgreSQL + Qdrant

Retrieve current financial policies, portfolio data and compliance documents.

Fine-tuned LLM

Base model + LoRA adapter

Apply learned classification, reasoning patterns and response formats.

Response generation

Retrieved context + fine-tuned behavior

Validation + security + human approval

Schema validation, source verification, business rules and HITL where required

### Example request flow

Suppose a customer asks:

> Should I increase my investment in a particular volatile asset?

The workflow would be:

1. FastAPI authenticates the customer and establishes trusted tenant context.
2. The RAG service retrieves the customer's authorized portfolio data.
3. It retrieves current investment policies and risk guidelines.
4. The fine-tuned model applies the learned financial risk classification and response structure.
5. The application validates the generated response.
6. The system verifies that the answer is grounded in retrieved information.
7. If the request requires a regulated or consequential decision, it routes it to an authorized human reviewer.
8. The final response includes citations to the retrieved documents.

The key distinction is that the fine-tuned model does not replace the retrieval layer or authorization layer.

You should never rely on fine-tuning to enforce tenant isolation, permissions, financial rules, or access control.

## 11. Common fine-tuning mistakes in production

These are useful interview discussion points.

| Mistake                                  | Why it is problematic                    | Better approach                         |
| ---------------------------------------- | ---------------------------------------- | --------------------------------------- |
| Training on poor-quality data            | Model learns incorrect patterns          | Expert-reviewed examples                |
| Training and testing on similar examples | Inflated evaluation scores               | Leakage-resistant data splits           |
| Fine-tuning to memorize policies         | Knowledge becomes stale                  | Use RAG for changing facts              |
| Training on sensitive customer records   | Privacy and memorization risks           | De-identification and approved datasets |
| Too many training epochs                 | Overfitting                              | Validation-based checkpoint selection   |
| Choosing LoRA rank arbitrarily           | Underfitting or unnecessary capacity     | Experiment with ranks                   |
| Evaluating only training loss            | Does not measure actual task performance | Held-out task evaluation                |
| Skipping safety evaluation               | Model may produce unsafe outputs         | Adversarial and policy testing          |

One additional consideration: fine-tuning is not automatically cheaper than using a hosted LLM. You must include GPU training, storage, inference infrastructure, adapter management, and model maintenance in the cost comparison.

## 12. How would you decide between RAG, fine-tuning, and prompt engineering?

Use this decision framework:

What problem are you trying to solve?

The model doesn't know the latest information.

Choose RAG

Example: Updated compliance policy or customer portfolio.

The model knows the task but behaves inconsistently.

Consider fine-tuning

Example: Standardized financial classification or output format.

The task needs better instructions or a few examples.

Start with prompt engineering

Example: Clarifying the output schema or response tone.

You need current knowledge and specialized behavior.

Combine RAG + fine-tuning

Example: Your enterprise financial advisor copilot.

My recommendation is to start with prompt engineering and RAG, establish an evaluation baseline, and fine-tune only when evaluation demonstrates a repeatable behavior gap that better prompts cannot economically solve.

## 13. Senior AI Engineer interview answer

If an interviewer asks, "How do you perform fine-tuning using LoRA or QLoRA, and when is it better than RAG?", you can answer:

> Fine-tuning adapts a pretrained language model using domain-specific input-output examples. Instead of updating every parameter, I prefer parameter-efficient fine-tuning techniques such as LoRA or QLoRA when they meet the quality requirements.
>
> LoRA freezes the pretrained model and introduces two trainable low-rank matrices, A and B, to approximate the weight update. This substantially reduces the number of trainable parameters.
>
> QLoRA takes this further by loading the base model in 4-bit quantization, commonly NF4, and training LoRA adapters while keeping the quantized base weights frozen. This reduces GPU memory requirements.
>
> My implementation would involve preparing a high-quality instruction dataset, splitting it into training, validation and test sets, loading the pretrained model, configuring LoRA adapters, training with supervised fine-tuning, and evaluating the resulting adapter on held-out data.
>
> I would measure task accuracy, precision, recall, output-schema validity, hallucination, safety, latency and inference cost. I would compare the fine-tuned model against the original model and a prompt-only baseline.
>
> Regarding RAG versus fine-tuning, I use RAG when the application needs frequently changing knowledge, private documents, source citations or customer-specific information. I use fine-tuning when I need consistent domain-specific behavior, classification, extraction or response formatting.
>
> For enterprise applications, I often combine both. RAG supplies the latest authorized knowledge, while the fine-tuned model learns the specialized task behavior.

## 14. What I would prepare next for a Staff AI Engineer interview

Make sure you can explain these five follow-up topics, because interviewers often go deeper after this question:

1. SFT vs DPO vs RLHF: How supervised fine-tuning differs from preference optimization and reinforcement learning.
2. LoRA rank selection: How to choose rank 8, 16, 32 or 64, and how to identify underfitting or overfitting.
3. Catastrophic forgetting: How fine-tuning can damage capabilities learned during pretraining and how to mitigate it.
4. Fine-tuning evaluation: How to build a reliable dataset, avoid data leakage, and compare base versus fine-tuned models.
5. Production deployment: How to version adapters, roll them back, serve multiple adapters, and monitor quality and GPU utilization.

The central Staff-level takeaway: Don't fine-tune just because you have domain-specific data. Fine-tune when you have evidence that the model needs to learn a more consistent behavior. Use retrieval when it needs access to information that changes. Use both when the application needs both capabilities.