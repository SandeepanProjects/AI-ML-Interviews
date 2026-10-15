# Preference Optimization in LLMs — From Fundamentals to Production AI Engineering

Preference optimization is a training approach that teaches an LLM which responses are better than other responses. Instead of giving the model only one correct answer, we provide comparisons such as:

- Response A is better than Response B.
- Response A follows the instructions more accurately.
- Response B contains unsupported claims.
- Response A is safer, clearer, or better structured.

The model uses these preferences to learn to produce responses that align more closely with the desired behavior.

For example, in your financial compliance AI platform, two answers might both sound reasonable, but one invents a policy citation while the other explicitly states that the available evidence is insufficient. Preference optimization can help train the model to favor the evidence-grounded response.

Let's understand the concepts, implement the core mathematics in PyTorch, and then look at how to fine-tune a real LLM using Direct Preference Optimization (DPO).

## 1. Why do we need preference optimization?

Suppose we have an instruction-tuned LLM and ask:

> Explain whether this investment advertisement complies with the applicable policy.

It generates two possible responses.

Response A — Preferred

"The claim may require review because the provided evidence does not substantiate the stated return. The applicable policy section should be verified before making a final compliance determination."

Cautious, evidence-aware, and does not invent a citation.

Response B — Rejected

"This advertisement is definitely compliant under Policy 7.4 and guarantees investors a 20% return."

Makes unsupported claims and fabricates a policy reference.

Instruction tuning can teach the model to produce an appropriate compliance-review response. Preference optimization adds another signal: given two candidate responses, which one should the model prefer?

That is useful because many real-world tasks do not have one uniquely correct wording. Multiple answers can be valid, but some are safer, more relevant, more faithful to evidence, or more helpful than others.

## 2. Where preference optimization fits in LLM training

1\. Pretraining

Learn general language patterns.

2\. Supervised fine-tuning (SFT)

Learn to follow instruction–response examples.

3\. Preference optimization

Learn which responses are preferred over alternatives.

4\. Evaluation and deployment

Validate quality, safety, regressions, and operational performance.

This is a common workflow, not a mandatory sequence. Some models use preference optimization without a separate SFT stage, and some use other alignment methods.

## 3. What does a preference dataset look like?

An ordinary supervised fine-tuning dataset contains an instruction and a target answer.

A preference dataset contains an instruction, a preferred answer, and a less-preferred answer.

```
{
  "prompt": "Explain why data leakage is dangerous.",
  "chosen": "Data leakage occurs when training uses information unavailable at prediction time. It can inflate validation scores and cause poor real-world performance.",
  "rejected": "Data leakage always improves a model and should be encouraged."
}
```

The training example tells the model:

- Given this prompt, the `chosen` answer is preferred.
- The `rejected` answer is less desirable.
- The model should learn to assign a higher relative preference to the chosen response.

A more realistic dataset can include comparisons about factual grounding, style, safety, completeness, instruction adherence, and tool-use decisions.

Critical distinction: a preference label is not necessarily a guarantee of factual correctness. If annotators prefer confident-sounding but incorrect responses, the model may learn that undesirable behavior. Preference data must be carefully designed and evaluated.

## 4. The main types of preference optimization

Several methods exist, but three are particularly important.

| Method | How it works                                                         | Engineering consideration                                   |
| ------ | -------------------------------------------------------------------- | ----------------------------------------------------------- |
| RLHF   | Uses a learned reward model and reinforcement learning, commonly PPO | More components and operational complexity                  |
| DPO    | Learns directly from chosen/rejected response pairs                  | Simpler training pipeline; still needs good preference data |
| ORPO   | Combines supervised learning and an odds-ratio preference objective  | Can reduce the need for a separate reference model          |

Other methods include IPO, KTO, and SimPO. Each has different objectives and assumptions.

We'll focus on DPO because it is a useful way to understand preference optimization mathematically and implement it in practice.

## 5. How DPO works

DPO stands for Direct Preference Optimization.

Its central idea is to increase the relative probability of a preferred response while decreasing the relative probability of a rejected response, compared with a reference model.

Imagine the prompt:

```
Explain RAG to a junior engineer.
```

The model generates two responses:

```
Chosen:
RAG retrieves relevant external information and supplies it
as context to a language model before it generates an answer.

Rejected:
RAG is a database that trains the LLM automatically.
```

DPO evaluates how the trainable policy model and the reference model score both responses.

Let:

- \\(\pi\_\theta\\) be the trainable policy model.
- \\(\pi\_{\text{ref}}\\) be the reference model, usually a frozen copy of the SFT model.
- \\(y_w\\) be the chosen (preferred) response.
- \\(y_l\\) be the rejected response.
- \\(\beta\\) be a parameter controlling the strength of the preference objective.

The DPO loss is:

\\[ \mathcal{L}\_{DPO} = -\log \sigma\left( \beta \left[ \log\frac{\pi\_\theta(y_w|x)} {\pi\_{\mathrm{ref}}(y_w|x)} - \log\frac{\pi\_\theta(y_l|x)} {\pi\_{\mathrm{ref}}(y_l|x)} \right]\right) \\]

Here, \\(\sigma\\) is the sigmoid function.

In practical terms:

1. Calculate the log-probability of the chosen response under the policy model.
2. Calculate the log-probability of the rejected response under the policy model.
3. Calculate the corresponding log-probabilities under the reference model.
4. Compare the policy's relative preference with the reference model's relative preference.
5. Optimize the loss to increase the desired preference margin.

The reference model matters because DPO does not simply say, "Make the chosen answer more probable at any cost." It optimizes preferences relative to a baseline model, reflecting a particular connection between reward optimization and KL-constrained policy optimization.

## 6. Understand the math with a numerical example

Suppose the model scores these responses as follows. The numbers are illustrative sequence log-probabilities.

| Response          | Reference model | Current policy |
| ----------------- | --------------- | -------------- |
| Chosen response   | -2.0            | -1.0           |
| Rejected response | -1.5            | -2.5           |

For the chosen response:

\\[ -1.0 - (-2.0) = 1.0 \\]

For the rejected response:

\\[ -2.5 - (-1.5) = -1.0 \\]

The relative preference margin is:

\\[ 1.0-(-1.0)=2.0 \\]

With \\(\beta=0.5\\), the sigmoid input is \\(1.0\\). Therefore, the loss is:

\\[ -\log(\sigma(1.0))\approx 0.313 \\]

A lower DPO loss indicates that the model has learned to favor the chosen response more strongly relative to the rejected response, according to this objective.

Notice that we are using sequence log-probabilities, not just the probability of the first token. In actual training, the response-token log-probabilities are accumulated according to the implementation's token masking and normalization conventions.

## 7. Implement the DPO loss from scratch in PyTorch

Let's implement the core mathematical objective without a language model first. This helps you understand what the trainer optimizes.

Install PyTorch:

```
pip install torch
```

Then run:

```
import torchimport torch.nn.functional as Fdef dpo_loss(    policy_chosen_logp,    policy_rejected_logp,    reference_chosen_logp,    reference_rejected_logp,    beta=0.1,):    """    Inputs are sequence-level log-probabilities.    Each tensor has shape [batch_size].    """    policy_log_ratio = (        policy_chosen_logp - policy_rejected_logp    )    reference_log_ratio = (        reference_chosen_logp - reference_rejected_logp    )    preference_margin = (        policy_log_ratio - reference_log_ratio    )    loss = -F.logsigmoid(beta * preference_margin)    return loss.mean()# Illustrative log-probabilities for one preference pair.policy_chosen = torch.tensor([-1.0], requires_grad=True)
```

### What should you understand from this code?

- `policy_chosen_logp` measures how likely the trainable model considers the chosen response.
- `policy_rejected_logp` measures how likely it considers the rejected response.
- The reference model supplies the baseline for the comparison.
- `F.logsigmoid` computes the numerically stable log-sigmoid used in the loss.
- `loss.backward()` computes gradients for these illustrative log-probabilities.

The gradients in this demonstration are not gradients of a real language model's parameters. They only show the direction of the loss with respect to the supplied scores. In a real DPO trainer, each sequence log-probability is calculated by running the policy and reference models over tokenized prompts and responses; backpropagation then updates the policy model's trainable parameters.

Also, this function assumes the inputs are already valid sequence-level log-probabilities. A real trainer must carefully handle padding, end-of-sequence tokens, prompt masking, and completion-token boundaries.

## 8. Fine-tune a real LLM using DPO

Now let's move from the mathematical demonstration to a practical training pipeline using Hugging Face TRL.

We'll use a small instruction-tuned model and LoRA adapters. This example demonstrates the workflow; the tiny dataset is for testing the pipeline, not for producing a production-quality model.

### Step 1: Create a preference dataset

Create `preferences.jsonl`:

```
{"prompt":"Explain RAG to a junior engineer.","chosen":"RAG retrieves relevant external information and provides it as context to an LLM before the model generates an answer.","rejected":"RAG is a database that automatically retrains the LLM whenever a document is added."}
{"prompt":"What should an assistant do when a policy citation cannot be verified?","chosen":"It should state that the citation could not be verified and request or retrieve authoritative policy evidence before making a definitive claim.","rejected":"It should invent a plausible policy number so that the answer looks complete."}
{"prompt":"Explain data leakage in machine learning.","chosen":"Data leakage occurs when training uses information that would not legitimately be available at prediction time, potentially inflating validation performance.","rejected":"Data leakage is always beneficial because it increases validation accuracy and makes a model more reliable."}
```

For each record:

- `prompt` is the input.
- `chosen` is the preferred completion.
- `rejected` is the less-preferred completion.

In a real project, use many diverse, quality-reviewed preference pairs. Include hard cases where both responses are plausible, and make sure the preferred response is actually better.

### Step 2: Install dependencies

```
pip install torch transformers datasets peft trl accelerate
```

### Step 3: Create the DPO training script

Save the following as `train_dpo.py`.

```
from datasets import load_datasetfrom transformers import AutoModelForCausalLM, AutoTokenizerfrom peft import LoraConfigfrom trl import DPOConfig, DPOTrainerBASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"DATA_PATH = "preferences.jsonl"OUTPUT_DIR = "./artifacts/dpo-adapter"# 1. Load the instruction-tuned starting model.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)# 2. Load preference pairs.dataset = load_dataset(    "json",    data_files=DATA_PATH,    split="train",)# 3. Format prompts using the model's chat template.def format_example(example):    messages = [        {            "role": "system",            "content": "You are a helpful, accurate technical assistant.",        },        {            "role": "user",            "content": example["prompt"],        },    ]    formatted_prompt = tokenizer.apply_chat_template(        messages,
```

Run it:

```
python train_dpo.py
```

Compatibility note: this uses the modern TRL `DPOTrainer`/`DPOConfig` API. Pin and test compatible library versions for a reproducible project. The model must support the specified LoRA target modules, and your hardware must have enough memory. The prompt formatting and completion boundaries should be validated against the tokenizer's actual chat template before training at scale.

### What does the trainer do internally?

Conceptually, it performs the following sequence for every batch:

Preference pairs

Prompt + chosen completion + rejected completion

Policy model

Scores both responses; gradients update trainable parameters.

Reference model

Provides baseline response scores; remains frozen.

DPO loss

Compute relative preference margin and optimize the policy.

Updated policy

Repeat for batches and epochs, then evaluate on held-out data.

With PEFT, TRL can use the base model with its adapter disabled as the reference behavior, rather than maintaining two fully separate trainable models. Exact reference-model handling depends on the trainer and PEFT configuration.

## 9. What does `beta` mean in DPO?

The `beta` parameter controls how strongly the preference margin affects the objective.

- A larger `beta` makes the sigmoid respond more sharply to differences in the relative preference margin.
- A smaller `beta` makes the objective less sensitive to that margin's scale.
- The learning rate controls the optimizer's update size; `beta` and learning rate are not interchangeable.

Do not assume that increasing `beta` always improves alignment. Its effects depend on the model, data, loss implementation, and training configuration. Tune it using validation results.

## 10. How is DPO different from RLHF?

Traditional RLHF often includes a separate reward-model training stage and a reinforcement-learning stage, such as PPO. DPO uses preference pairs to optimize the policy directly without requiring that separate reward-model-plus-RL loop.

RLHF with a reward model

Preference data → Train reward model → Optimize policy with RL → Evaluate

DPO

Preference pairs → Compute DPO objective using policy and reference scores → Update policy → Evaluate

DPO is simpler in many training setups, but it is not automatically cheaper in every case. It still requires multiple sequence-scoring computations, careful data preparation, sufficient training resources, and strong evaluation.

## 11. How would you use preference optimization in your financial compliance AI project?

For your compliance-review platform, I would use a preference dataset designed around evidence-grounded behavior.

| Scenario                      | Preferred behavior                                               |
| ----------------------------- | ---------------------------------------------------------------- |
| Missing policy evidence       | Acknowledge uncertainty instead of inventing a citation          |
| Unsupported performance claim | Explain the evidence gap and flag it for review                  |
| Ambiguous policy              | State the ambiguity and route the case to an authorized reviewer |
| Structured review             | Follow the required JSON schema                                  |
| Sensitive request             | Respect authorization and safety constraints                     |

The production pipeline would look like this:

1. Collect representative prompts and candidate responses.
2. Have qualified reviewers compare responses against documented criteria.
3. Remove sensitive data where appropriate and maintain dataset provenance.
4. Train an SFT baseline and then evaluate whether DPO adds measurable value.
5. Compare both models on held-out quality, factual grounding, safety, and regression tests.
6. Deploy the selected model behind existing authorization, retrieval, validation, and human-review controls.

DPO should not be used to teach the model that it can ignore missing evidence or bypass compliance checks. Those controls must remain enforced in the application architecture.

## 12. Senior/staff engineer checklist

Before approving a preference-optimization project, verify the following:

- Data quality: Are chosen responses consistently better according to documented criteria?
- Leakage prevention: Are train and test examples separated by source, customer, or document where necessary?
- Training correctness: Are prompt tokens, completion tokens, padding, and EOS handling correct?
- Baseline: Does the DPO model outperform the SFT baseline on held-out tests?
- Safety: Are refusal behavior, unsupported claims, and prompt-injection scenarios evaluated?
- Regression: Has general capability been checked so that improving one task doesn't degrade others?
- Reproducibility: Are the base model, adapter, dataset version, code, and hyperparameters recorded?
- Deployment: Can the team roll back to the previous model if evaluation or production monitoring detects a regression?

The most important takeaway is that instruction tuning teaches the model what a good answer looks like; preference optimization teaches it to favor better answers over worse alternatives. Neither technique guarantees factual correctness, so production systems still need evidence retrieval, independent validation, access controls, and evaluation.