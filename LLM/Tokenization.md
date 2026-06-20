This is one of the **most frequently asked LLM interview topics** for Senior AI/ML Engineer, Applied Scientist, and Generative AI Engineer roles.

Most candidates say:

> "Tokenization converts text into tokens."

That answer is not enough.

A senior engineer should be able to explain:

* Why tokenization exists
* Why words are not directly used
* How BPE works internally
* How WordPiece works internally
* How SentencePiece works internally
* Why GPT uses BPE
* Why BERT uses WordPiece
* Why Llama uses SentencePiece
* How to implement them from scratch

---

# Why Do We Need Tokenization?

Suppose we have a Transformer.

The input is

```text
I love deep learning
```

Can a Transformer understand text?

No.

Neural networks only understand numbers.

So first we must convert

```text
"I love AI"
```

into

```text
[15, 203, 98]
```

These numbers are called **Token IDs**.

The complete flow is:

```text
Text
   │
   ▼
Tokenizer
   │
   ▼
Token IDs
   │
   ▼
Embedding Layer
   │
   ▼
Vectors
   │
   ▼
Transformer
```

Without tokenization, the embedding layer cannot look up vectors.

---

# Why Not Store Every Word?

Suppose English has

```text
Dog
Dogs
Dog's
Dogging
Dogged
```

If every variation is stored separately:

```text
Vocabulary

Dog

Dogs

Dogging

Dogged

Dog's
```

Now imagine millions of words across many languages.

Vocabulary becomes enormous.

Problems:

* Huge embedding matrix
* High memory usage
* Unknown words cannot be handled

Example

```text
Supercalifragilisticexpialidocious
```

Never seen during training.

Normal word-level tokenizer:

```text
UNKNOWN
```

LLMs cannot afford this.

Need smaller reusable pieces.

---

# Subword Tokenization

Instead of

```text
Internationalization
```

Store

```text
International

ization
```

or

```text
Inter

nation

al

ization
```

Now even unseen words can be built.

Example

```text
Cybersecurity
```

↓

```text
Cyber

Security
```

Even if never seen before.

This is why modern LLMs use **subword tokenization**.

---

# Types of Tokenization

| Method        | Used By                |
| ------------- | ---------------------- |
| Character     | Early RNNs             |
| Word          | Classical NLP          |
| BPE           | GPT, GPT-2, GPT-3      |
| WordPiece     | BERT                   |
| SentencePiece | Llama, T5, ALBERT, mT5 |

Let's build them one by one.

---

# 1. Byte Pair Encoding (BPE)

This is the tokenizer used by GPT-family models.

Idea:

Start with characters.

Repeatedly merge the most frequent adjacent pair.

Eventually frequent words become single tokens.

---

Imagine training corpus

```text
low

lower

lowest

newest

widest
```

Initially

Every word splits into characters.

```text
l o w

l o w e r

l o w e s t
```

Vocabulary

```text
l

o

w

e

r

s

t
```

---

## Count Adjacent Pairs

We count

```text
lo

ow

we

er

es

st
```

Suppose

```text
lo → 500

ow → 490

we → 80

er → 60
```

Most common pair

```text
lo
```

Merge it.

Vocabulary

```text
lo

w

e

r

s

t
```

---

Repeat

Now

```text
lo w
```

becomes

```text
low
```

Eventually vocabulary

```text
low

lower

lowest

newest
```

Frequent words become one token.

Rare words remain multiple pieces.

---

Example

```text
unhappiness
```

may become

```text
un

happi

ness
```

instead of

```text
u

n

h

a

p

p

i

...
```

---

## BPE Algorithm

```text
Characters

↓

Find Most Frequent Pair

↓

Merge

↓

Repeat 50,000 Times

↓

Final Vocabulary
```

---

## BPE From Scratch

```python
from collections import Counter

words = [
    "low",
    "lower",
    "lowest"
]

tokens = [list(word) for word in words]

pair_counts = Counter()

for word in tokens:
    for i in range(len(word)-1):
        pair = (
            word[i],
            word[i+1]
        )
        pair_counts[pair]+=1

print(pair_counts)
```

Output

```text
('l','o'):3

('o','w'):3

('w','e'):2
```

Merge

```text
l + o

↓

lo
```

Repeat thousands of times.

---

## Why GPT Uses BPE

Advantages

* Small vocabulary
* Fast
* Handles unknown words
* Excellent compression
* Language independent

---

# 2. WordPiece

Used in

* BERT
* DistilBERT
* ELECTRA

Looks similar to BPE.

But internally different.

---

## BPE

Chooses

Most Frequent Pair.

Example

```text
ab →100

bc →80
```

Merge

```text
ab
```

Simply because frequency highest.

---

## WordPiece

Chooses

Most Useful Pair.

Not just frequent.

It maximizes the likelihood of the training data under a language model objective. In practice, it favors merges that provide the greatest improvement rather than simply occurring most often.

Very important interview point.

---

Suppose

```text
playing

player

played
```

Instead of merging

```text
pl
```

It may merge

```text
play
```

because

```text
play

+

ing

+

er

+

ed
```

represents many words efficiently.

---

Vocabulary

```text
play

##ing

##er

##ed
```

Notice

```text
##
```

Means

Continuation token.

---

Example

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

## WordPiece Example

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

tokens = tokenizer.tokenize(
    "playing unbelievable"
)

print(tokens)
```

Output

```text
['playing',
 'un',
 '##believable']
```

or for other words

```text
play

##ing
```

depending on vocabulary.

---

## Why BERT Uses WordPiece

Goal

Masked Language Modeling.

Need meaningful word fragments.

WordPiece generally produces semantically useful pieces.

---

# 3. SentencePiece

Used by

* Llama
* T5
* ALBERT
* mT5
* Many multilingual LLMs

SentencePiece is very different.

---

## Biggest Difference

BPE and WordPiece

Need spaces.

Sentence

```text
I love AI
```

↓

```text
I

love

AI
```

SentencePiece

Does NOT rely on whitespace.

Instead

Entire sentence

is treated as a raw byte stream.

Spaces become normal symbols.

Example

```text
I love AI
```

↓

```text
▁I

▁love

▁AI
```

The special symbol

```text
▁
```

means

Start of a new word.

---

Why?

Japanese

Chinese

Thai

do not separate words using spaces.

SentencePiece works across languages.

---

## SentencePiece Training

Corpus

```text
I love AI

I love ML
```

Model learns

```text
▁I

▁love

▁AI

▁ML
```

Entirely from raw text.

No pre-tokenization.

---

## Code

```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor()

sp.load("tokenizer.model")

tokens = sp.encode(
    "I love transformers",
    out_type=str
)

print(tokens)
```

Output

```text
['▁I',
 '▁love',
 '▁transform',
 'ers']
```

---

## Why Llama Uses SentencePiece

Advantages

* Language independent
* No whitespace assumptions
* Excellent multilingual support
* Handles Unicode naturally
* Robust for unseen words

---

# Internal Comparison

Suppose

```text
internationalization
```

### BPE

```text
international

ization
```

or

```text
inter

national

ization
```

depending on learned merges.

---

### WordPiece

```text
international

##ization
```

Uses continuation markers.

---

### SentencePiece

```text
▁international

ization
```

or another learned segmentation that includes the leading-word marker.

---

# Training Flow

### BPE

```text
Corpus
      │
      ▼
Split into Characters
      │
      ▼
Count Pair Frequencies
      │
      ▼
Merge Most Frequent Pair
      │
      ▼
Repeat Until Vocabulary Size
```

---

### WordPiece

```text
Corpus
      │
      ▼
Split into Characters
      │
      ▼
Evaluate Candidate Merges
      │
      ▼
Choose Merge That Best Improves
Language Model Likelihood
      │
      ▼
Repeat
```

---

### SentencePiece

```text
Raw Text
      │
      ▼
No Pre-tokenization
      │
      ▼
Treat Space as Symbol (▁)
      │
      ▼
Learn Subword Vocabulary
      │
      ▼
Encode Sentences
```

---

# Comparison Table

| Feature                             | BPE                                          | WordPiece                   | SentencePiece                                           |
| ----------------------------------- | -------------------------------------------- | --------------------------- | ------------------------------------------------------- |
| Used By                             | GPT, GPT-2, GPT-3                            | BERT, DistilBERT            | Llama, T5, ALBERT                                       |
| Initial Unit                        | Characters (or bytes in byte-level variants) | Characters                  | Raw text                                                |
| Merge Rule                          | Most frequent pair                           | Best likelihood improvement | Learns subwords from raw text (BPE or Unigram variants) |
| Continuation Marker                 | No                                           | `##`                        | `▁` marks word start                                    |
| Requires Pre-tokenization by Spaces | Usually yes (classic BPE)                    | Usually yes                 | No                                                      |
| Multilingual Support                | Good                                         | Good                        | Excellent                                               |
| Handles Unknown Words               | Yes (via subwords)                           | Yes                         | Yes                                                     |
| Best For                            | Autoregressive LLMs                          | Masked language models      | Multilingual and modern LLMs                            |

---

# Senior AI Engineer Interview Answer (2–3 minutes)

> "Modern Transformers don't operate directly on words; they operate on token IDs produced by a tokenizer. Since word-level vocabularies become extremely large and cannot handle unseen words effectively, modern LLMs use subword tokenization. BPE, used in GPT models, iteratively merges the most frequent adjacent symbol pairs to build an efficient vocabulary. WordPiece, used in BERT, chooses merges based on how much they improve the language model's likelihood, producing semantically meaningful subwords marked with `##`. SentencePiece, used in Llama and T5, eliminates the need for whitespace-based pre-tokenization by treating the entire input as raw text and representing word boundaries with the `▁` symbol. These approaches reduce vocabulary size, improve handling of rare and unseen words, and make Transformer models more efficient and robust across languages.
