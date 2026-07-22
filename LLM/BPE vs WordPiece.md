# BPE vs WordPiece (Senior AI Engineer Level)

Both **BPE (Byte Pair Encoding)** and **WordPiece** are **subword tokenization algorithms** used to convert text into tokens before feeding it to a Transformer.

Instead of splitting only by words, they split words into **subwords**, allowing models to:

* Handle unknown words
* Reduce vocabulary size
* Learn meaningful word pieces
* Work with multiple languages

---

# Why Do We Need Subword Tokenization?

Suppose the sentence is:

```text
Artificial Intelligence is transforming healthcare.
```

If the vocabulary contains only:

```text
Artificial
Intelligence
is
transforming
```

then the word:

```text
healthcare
```

may be unknown.

Instead of producing:

```text
<UNK>
```

the tokenizer splits it into:

```text
health
care
```

or

```text
health
##care
```

The model can still understand the word.

---

# High-Level Comparison

| Feature            | BPE                   | WordPiece                                    |
| ------------------ | --------------------- | -------------------------------------------- |
| Introduced by      | Compression algorithm | Google                                       |
| Used in            | GPT-2, RoBERTa        | BERT                                         |
| Merge criterion    | Most frequent pair    | Highest probability (likelihood improvement) |
| Training objective | Frequency-based       | Language-model-based                         |
| Unknown words      | Rare                  | Rare                                         |
| Speed              | Faster                | Slightly slower                              |
| Vocabulary quality | Good                  | Often more linguistically meaningful         |

---

# Byte Pair Encoding (BPE)

## Idea

Repeatedly merge the **most frequent adjacent pair of symbols**.

Start with characters.

Example vocabulary:

```text
low
lowest
new
newest
```

Initial representation:

```text
l o w

l o w e s t

n e w

n e w e s t
```

---

## Count Adjacent Pairs

Suppose:

```text
lo

15

ow

15

es

20

st

20
```

The most frequent pair is:

```text
es
```

Merge:

```text
e s

↓

es
```

Vocabulary becomes:

```text
l o w

l o w es t

n e w

n e w es t
```

Repeat.

Next merge:

```text
es + t

↓

est
```

Eventually:

```text
lowest

↓

low est
```

---

## BPE Training Algorithm

```text
Characters

↓

Count adjacent pairs

↓

Find most frequent pair

↓

Merge pair

↓

Repeat N times
```

---

## Python Example

```python
from collections import Counter

words = [
    "low",
    "lowest",
    "new",
    "newest"
]

pairs = Counter()

for word in words:
    chars = list(word)
    for i in range(len(chars)-1):
        pairs[(chars[i], chars[i+1])] += 1

print(pairs)
```

Output:

```text
('e','s'):2

('s','t'):2

('l','o'):2

...
```

Merge the most frequent pair and continue.

---

# WordPiece

Google developed WordPiece for BERT.

Unlike BPE, WordPiece does **not** simply merge the most frequent pair.

Instead it chooses the merge that **maximizes the probability of the training data** (i.e., provides the greatest improvement to the language model objective).

This usually produces a more efficient vocabulary.

---

## Example

Vocabulary:

```text
play

playing

played

player
```

Candidate merges:

```text
play

playing

played

player
```

WordPiece prefers merges that maximize the likelihood of the corpus rather than just raw frequency.

Final vocabulary may be:

```text
play

##ing

##ed

##er
```

---

# WordPiece Uses Prefixes

Example:

```text
playing
```

becomes:

```text
play

##ing
```

The prefix:

```text
##
```

means:

```text
This token continues the previous word.
```

Example:

```text
unbelievable
```

↓

```text
un

##believ

##able
```

---

# BPE Example

Suppose:

```text
playing
```

BPE may tokenize as:

```text
play

ing
```

No continuation marker is required.

---

# Tokenization Comparison

Sentence:

```text
unhappiness
```

### BPE

```text
un

happi

ness
```

### WordPiece

```text
un

##happy

##ness
```

---

# Unknown Word Example

Sentence:

```text
electroencephalography
```

### Character tokenizer

```text
e

l

e

c

t

...
```

Too many tokens.

### Word tokenizer

```text
<UNK>
```

Fails completely.

### BPE

```text
electro

encephalo

graphy
```

### WordPiece

```text
electro

##encephalo

##graphy
```

---

# Vocabulary Building

## BPE

```text
Start

a

b

c

d

↓

Merge most frequent

ab

↓

Merge again

abc

↓

Merge again

abcd
```

Frequency only.

---

## WordPiece

```text
Start

Characters

↓

Evaluate candidate merges

↓

Compute likelihood improvement

↓

Choose best merge

↓

Repeat
```

---

# Training Objective

## BPE

Chooses:

```text
Most Frequent Pair
```

Example:

```text
th

appears

1000 times
```

Merge:

```text
t+h

↓

th
```

---

## WordPiece

Chooses:

```text
Merge giving maximum probability
```

Example:

Even if:

```text
th

appears

1000 times
```

another merge may be preferred if it results in a vocabulary that better models the corpus.

---

# Real Models

| Model      | Tokenizer                   |
| ---------- | --------------------------- |
| GPT-2      | BPE                         |
| RoBERTa    | BPE                         |
| BART       | BPE                         |
| BERT       | WordPiece                   |
| ALBERT     | WordPiece                   |
| DistilBERT | WordPiece                   |
| LLaMA      | SentencePiece (BPE variant) |
| T5         | SentencePiece (Unigram)     |

---

# BPE Training Complexity

```text
Characters

↓

Count frequencies

↓

Merge highest pair

↓

Repeat 30k–50k merges
```

Very efficient.

---

# WordPiece Complexity

```text
Characters

↓

Evaluate many candidate merges

↓

Compute likelihood

↓

Choose best merge

↓

Repeat
```

Slightly slower during vocabulary training, but inference tokenization is still fast.

---

# Advantages

## BPE

✅ Simple

✅ Fast

✅ Easy to implement

✅ Good compression

---

## WordPiece

✅ Better vocabulary quality

✅ More linguistically meaningful splits

✅ Better handling of rare words

---

# Which One Is Better?

It depends on the goal:

* **BPE** is straightforward, efficient, and works very well for autoregressive language models like GPT.
* **WordPiece** optimizes a language-model objective during vocabulary construction, which often yields more meaningful subword units for encoder models like BERT.

In practice, both perform well, and modern tokenizers such as **SentencePiece** and **Unigram Language Model** have become increasingly popular because they simplify multilingual training and remove the need for language-specific pre-tokenization.

---

# Interview Questions

1. Why do Transformers use subword tokenization instead of word-level tokenization?
2. What problem does BPE solve?
3. How does BPE decide which tokens to merge?
4. How does WordPiece differ from BPE during vocabulary construction?
5. Why does WordPiece use the `##` prefix?
6. Why are subword tokenizers better than character-level tokenizers?
7. Which tokenizer is used by GPT, BERT, LLaMA, and T5?
8. What are the trade-offs between BPE, WordPiece, and SentencePiece?
9. Why is `<UNK>` much less common with subword tokenization?
10. How does the choice of tokenizer affect vocabulary size, sequence length, and model efficiency?
