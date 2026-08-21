For a **Senior AI Engineer / LLM Engineer interview**, fine-tuning questions usually go from fundamentals → implementation → optimization → evaluation → production.

Here is a strong interview question bank.

## 1. Fine-tuning fundamentals

1. **What is fine-tuning of an LLM?**
2. **Why would you fine-tune an LLM instead of using prompting?**
3. **Fine-tuning vs RAG — when would you use each?**
4. **Fine-tuning vs prompt engineering?**
5. **What is the difference between pre-training, fine-tuning, and inference?**
6. **What types of tasks are suitable for fine-tuning?**
7. **When should you NOT fine-tune an LLM?**
8. **What kind of dataset is required for fine-tuning?**
9. **How much training data is typically needed?**
10. **What happens internally during fine-tuning?**

---

# 2. Types of fine-tuning

11. **What is supervised fine-tuning (SFT)?**
12. **What is instruction tuning?**
13. **What is parameter-efficient fine-tuning (PEFT)?**
14. **What is LoRA?**
15. **What is QLoRA?**
16. **What is full fine-tuning?**
17. **LoRA vs full fine-tuning?**
18. **LoRA vs QLoRA?**
19. **What is adapter-based fine-tuning?**
20. **What is prefix tuning?**
21. **What is prompt tuning?**
22. **What is P-tuning?**
23. **What is the difference between PEFT and full fine-tuning?**

A very common interview question:

> **Why would you choose LoRA instead of full fine-tuning?**

A good answer should mention:

* fewer trainable parameters
* lower GPU memory
* faster training
* smaller checkpoints
* easier experimentation
* reduced risk of catastrophic forgetting
* adapters can be swapped for different tasks

---

# 3. LoRA — very important

24. **Explain LoRA mathematically.**
25. **Why does LoRA reduce the number of trainable parameters?**
26. **What are LoRA rank `r`, alpha, and dropout?**
27. **Which layers should you apply LoRA to?**
28. **Why are attention projections commonly targeted by LoRA?**
29. **What happens if you increase LoRA rank?**
30. **How do you choose the LoRA rank?**
31. **What is LoRA scaling?**
32. **What is the difference between LoRA adapters and the base model?**
33. **Can you merge LoRA weights into the base model?**
34. **What are the advantages and disadvantages of merging adapters?**

You should be able to explain:

[
W' = W + \Delta W
]

and LoRA approximates:

[
\Delta W = BA
]

where `A` and `B` are low-rank matrices.

---

# 4. QLoRA

35. **What is QLoRA?**
36. **Why was QLoRA introduced?**
37. **How does QLoRA reduce GPU memory?**
38. **What is 4-bit quantization?**
39. **What is NF4?**
40. **What is double quantization?**
41. **What is paged optimizers?**
42. **Can you fine-tune a 70B model using QLoRA? How?**
43. **What is the difference between QLoRA and LoRA?**
44. **Does QLoRA modify the quantized base model?**
45. **What precision are the LoRA adapters trained in?**

This is particularly important for senior interviews.

---

# 5. Dataset questions

46. **How would you create a fine-tuning dataset?**
47. **What format should an instruction-tuning dataset have?**
48. **What is conversational fine-tuning?**
49. **How do you clean a fine-tuning dataset?**
50. **How do you remove duplicate examples?**
51. **How do you detect bad training examples?**
52. **How do you handle noisy labels?**
53. **How do you handle class imbalance?**
54. **How do you split training and validation datasets?**
55. **How do you prevent data leakage?**
56. **Should the validation set contain examples similar to training data?**
57. **How do you handle long-context training examples?**
58. **How do you determine the maximum sequence length?**

A senior-level question:

> **Suppose you have 1 million customer-support conversations. How would you prepare them for fine-tuning?**

You should discuss:

```text
Raw conversations
       ↓
PII removal
       ↓
Quality filtering
       ↓
Deduplication
       ↓
Conversation normalization
       ↓
Instruction/response formatting
       ↓
Train / validation / test split
       ↓
Tokenization
       ↓
Fine-tuning
```

---

# 6. Training hyperparameters

59. **What is learning rate?**
60. **How do you choose learning rate for fine-tuning?**
61. **What is batch size?**
62. **What is gradient accumulation?**
63. **What is the relationship between batch size and learning rate?**
64. **What is number of epochs?**
65. **How many epochs should you use for LLM fine-tuning?**
66. **What is warmup?**
67. **What is weight decay?**
68. **What is gradient clipping?**
69. **What is cosine learning-rate scheduling?**
70. **What is linear learning-rate decay?**
71. **What is mixed precision training?**
72. **FP16 vs BF16?**
73. **What is gradient checkpointing?**
74. **How does gradient checkpointing reduce memory?**

---

# 7. Overfitting and catastrophic forgetting

75. **What is overfitting during LLM fine-tuning?**
76. **How do you detect overfitting?**
77. **How do you prevent overfitting?**
78. **What is catastrophic forgetting?**
79. **Why can fine-tuning cause catastrophic forgetting?**
80. **How would you preserve the model's general capabilities?**
81. **How does LoRA help reduce catastrophic forgetting?**
82. **What happens if you train for too many epochs?**
83. **What happens if the learning rate is too high?**
84. **What happens if the learning rate is too low?**

A strong answer should mention:

```text
Overfitting
    ↓
Training loss decreases
Validation loss increases
    ↓
Model memorizes training data
    ↓
Poor generalization
```

---

# 8. Instruction tuning

85. **What is instruction tuning?**
86. **How is instruction tuning different from traditional supervised learning?**
87. **What does an instruction dataset look like?**
88. **How would you fine-tune an LLM to behave like a customer-support agent?**
89. **How would you teach an LLM to output structured JSON?**
90. **How would you fine-tune a model for SQL generation?**
91. **How would you fine-tune a model for coding?**
92. **How would you fine-tune a model for a specific company's terminology?**

Example:

```json
{
  "instruction": "Summarize the customer's issue",
  "input": "My payment was deducted but the order failed.",
  "output": "The customer reports a failed order despite successful payment deduction."
}
```

---

# 9. Fine-tuning vs RAG

This is **extremely common in AI interviews**.

93. **When would you use RAG instead of fine-tuning?**
94. **When would you use fine-tuning instead of RAG?**
95. **Can fine-tuning teach an LLM private company knowledge?**
96. **Why is fine-tuning generally not a good replacement for a knowledge base?**
97. **Can you combine RAG and fine-tuning?**
98. **How would you build a system using both?**

A good architecture is:

```text
                 User Query
                     │
                     ▼
              Fine-tuned LLM
                     │
                     │
              Retrieval Layer
                     │
                     ▼
            Vector Database
                     │
                     ▼
              Relevant Context
                     │
                     ▼
              Fine-tuned LLM
                     │
                     ▼
                  Answer
```

Use **RAG for changing/private knowledge** and fine-tuning for **behavior, style, task specialization, and output patterns**.

---

# 10. RLHF and preference tuning

99. **What is RLHF?**
100. **Why is RLHF used?**
101. **What is reward modeling?**
102. **What is PPO?**
103. **What is DPO?**
104. **DPO vs RLHF?**
105. **What is preference tuning?**
106. **What is a preference dataset?**
107. **What is chosen vs rejected response?**
108. **Why is DPO easier to train than PPO?**
109. **What is ORPO?**
110. **What is GRPO?**
111. **When would you use SFT vs DPO?**

Typical preference data:

```json
{
  "prompt": "Explain recursion",
  "chosen": "Recursion is a technique where...",
  "rejected": "Recursion is when a function loops..."
}
```

---

# 11. Fine-tuning with Hugging Face

112. **How would you fine-tune Llama using Hugging Face?**
113. **What is `Trainer`?**
114. **What is `SFTTrainer`?**
115. **What is PEFT?**
116. **How do you configure LoRA using PEFT?**
117. **How do you load a quantized model?**
118. **How do you save LoRA adapters?**
119. **How do you load the adapters during inference?**
120. **How do you merge LoRA adapters?**
121. **How do you monitor training loss?**

Typical stack:

```text
Transformers
      +
Datasets
      +
PEFT
      +
TRL
      +
bitsandbytes
      +
PyTorch
```

---

# 12. Practical coding questions

You may be asked to:

122. **Write code to fine-tune an LLM using LoRA.**
123. **Write a QLoRA training script.**
124. **Prepare an instruction dataset.**
125. **Implement tokenization.**
126. **Configure `LoraConfig`.**
127. **Configure training arguments.**
128. **Load a 4-bit quantized model.**
129. **Save and reload LoRA adapters.**
130. **Run inference using a fine-tuned model.**
131. **Calculate training loss.**
132. **Implement evaluation after fine-tuning.**

---

# 13. Evaluation questions

133. **How do you evaluate a fine-tuned LLM?**
134. **What metrics do you use?**
135. **What is perplexity?**
136. **What is BLEU?**
137. **What is ROUGE?**
138. **What is BERTScore?**
139. **How do you evaluate instruction following?**
140. **How do you evaluate hallucinations?**
141. **How do you perform human evaluation?**
142. **What is LLM-as-a-judge?**
143. **How do you compare the base model against the fine-tuned model?**
144. **How would you perform A/B testing?**

For production, don't rely only on loss.

Think:

```text
                    Evaluation
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
     Offline          Human          Online
     metrics         evaluation       metrics
        │               │               │
   Loss/Perplexity   Quality         CTR
   ROUGE/BLEU        Relevance       Conversion
   Accuracy          Safety          User feedback
```

---

# 14. Production fine-tuning

145. **How would you fine-tune an LLM in production?**
146. **How would you manage training datasets?**
147. **How would you version models?**
148. **How would you version datasets?**
149. **How would you track experiments?**
150. **How would you monitor fine-tuning jobs?**
151. **How would you deploy a fine-tuned model?**
152. **How would you perform rollback?**
153. **How would you perform A/B testing?**
154. **How would you reduce inference cost?**
155. **How would you serve a LoRA model?**
156. **Can multiple LoRA adapters share one base model?**
157. **How would you handle multiple customers with different adapters?**

For your **multi-tenant AI platform**, this is a particularly good architecture question:

```text
                  API Gateway
                       │
                 Tenant ID
                       │
                       ▼
              Model Router
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
      Base Model    LoRA A       LoRA B
                    Tenant A     Tenant B
```

---

# 15. GPU / distributed training questions

158. **How much GPU memory is required to fine-tune a 7B model?**
159. **Why does full fine-tuning require so much memory?**
160. **How does LoRA reduce GPU requirements?**
161. **What is distributed data parallelism?**
162. **What is FSDP?**
163. **What is DeepSpeed?**
164. **What is ZeRO?**
165. **ZeRO-1 vs ZeRO-2 vs ZeRO-3?**
166. **How would you fine-tune a 70B model?**
167. **How would you handle out-of-memory errors?**
168. **How would you optimize training throughput?**

---

# 16. Advanced senior-level questions

These are the ones I'd prioritize for a **Senior AI Engineer interview**.

169. **How would you fine-tune a 7B model with only one 24GB GPU?**

170. **How would you fine-tune a 70B model with limited GPU resources?**

171. **How would you decide between full fine-tuning, LoRA and QLoRA?**

172. **How would you select LoRA target modules?**

173. **How would you determine the optimal LoRA rank?**

174. **How would you diagnose a fine-tuning run whose training loss decreases but validation performance gets worse?**

175. **How would you prevent catastrophic forgetting?**

176. **How would you build a dataset for domain-specific fine-tuning?**

177. **How would you remove PII from training data?**

178. **How would you detect data leakage?**

179. **How would you compare two fine-tuned models?**

180. **How would you deploy multiple LoRA adapters in production?**

181. **How would you fine-tune a model for structured JSON output?**

182. **How would you fine-tune a model for tool calling?**

183. **How would you fine-tune a model for function calling?**

184. **How would you fine-tune an LLM while preserving its general reasoning capability?**

185. **How would you combine SFT + DPO + RAG into one production architecture?**

---

## ⭐ The 15 questions I would prepare first

Given your **Senior AI Engineer interview preparation**, don't try to memorize all 185 initially. Master these:

1. **What is LLM fine-tuning?**
2. **Fine-tuning vs RAG**
3. **Full fine-tuning vs LoRA**
4. **Explain LoRA mathematically**
5. **What is QLoRA?**
6. **LoRA vs QLoRA**
7. **How do you prepare a fine-tuning dataset?**
8. **How do you prevent overfitting?**
9. **What is catastrophic forgetting?**
10. **How do you choose learning rate, rank, batch size and epochs?**
11. **How do you evaluate a fine-tuned LLM?**
12. **SFT vs DPO vs RLHF**
13. **How would you fine-tune a 7B/70B model with limited GPUs?**
14. **How would you deploy LoRA adapters in production?**
15. **Design a production fine-tuning pipeline end-to-end.**

The **last question is especially important** because it lets you demonstrate senior-level architecture rather than just knowing terminology:

```text
                  Data Sources
                       │
                       ▼
              Data Processing
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
        PII Removal          Quality Filter
             │                   │
             └─────────┬─────────┘
                       ▼
                 Dataset Version
                       │
                       ▼
              Train / Val / Test
                       │
                       ▼
              Tokenization
                       │
                       ▼
             SFT / LoRA / QLoRA
                       │
                       ▼
                Evaluation
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
       Quality Good           Quality Bad
            │                     │
            ▼                     └──► Iterate
       Model Registry
            │
            ▼
       Deployment
            │
            ▼
      A/B / Canary Test
            │
            ▼
        Monitoring
       ┌────┼────┐
       ▼    ▼    ▼
     Cost  Latency Quality
```

If you're preparing specifically for **Senior AI Engineer interviews**, the next step should be to learn **LoRA + QLoRA deeply with actual Hugging Face/PEFT code**, because interviewers commonly move from *“What is LoRA?”* to *“Show me how you'd implement it and explain every parameter.”*
