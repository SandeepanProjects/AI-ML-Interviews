# How would you remove PII from training data?

PII (**Personally Identifiable Information**) removal is an important part of preparing datasets for LLM fine-tuning.

A strong interview answer is:

> **I would build a multi-stage PII detection and sanitization pipeline rather than relying on regex alone. I would first classify the data, detect structured PII with regex and validation rules, detect contextual PII using NER or a PII detection service, replace or remove detected values, preserve safe placeholders when semantic structure matters, run a second validation scan, and keep audit metrics without storing the original sensitive values. For highly sensitive data, I would minimize collection and prefer de-identification before the data enters the training pipeline.**

---

# 1. What counts as PII?

Typical examples include:

```text
Name:
John Smith

Email:
john.smith@example.com

Phone:
+91 9876543210

Address:
123 MG Road, Bangalore

Date of birth:
01-05-1989

Government ID:
PAN / Aadhaar / Passport / SSN

Account number:
123456789

IP address:
192.168.1.10
```

For an enterprise dataset, you may also treat these as sensitive:

```text
Customer IDs
Employee IDs
Internal usernames
API keys
Passwords
Access tokens
Database credentials
Credit card numbers
```

Not everything is legally PII in every jurisdiction, but from an ML security perspective, sensitive identifiers should generally be handled explicitly.

---

# 2. Why remove PII before fine-tuning?

Suppose the training data contains:

```text
Customer: John Smith
Email: john.smith@example.com

Issue:
Customer cannot access their account.
```

The model can potentially learn associations from this information.

The risks include:

```text
Training data
      ↓
Model learns patterns
      ↓
Possible memorization
      ↓
Sensitive information exposure
```

You want:

```text
Original data
      ↓
PII detection
      ↓
PII transformation
      ↓
Validation scan
      ↓
Safe training dataset
```

---

# 3. The production architecture

I would design something like:

```text
Raw Data
   │
   ▼
Data Access Control
   │
   ▼
PII Scanner #1
   │
   ├── Regex
   ├── Validators
   ├── NER
   └── PII detection model/service
   │
   ▼
PII Transformation
   │
   ├── Remove
   ├── Mask
   ├── Replace with placeholder
   └── Reject record
   │
   ▼
PII Scanner #2
   │
   ▼
Quality Validation
   │
   ▼
Approved Training Dataset
```

The second scan is important because the first detection stage can miss data.

---

# 4. Method 1: Regex for structured PII

Regex works well for predictable formats.

Examples:

* Email
* Phone numbers
* Credit card-like numbers
* IP addresses

## Python example

```python
import re


PII_PATTERNS = {
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",

    "PHONE": (
        r"\b(?:\+91[-\s]?)?"
        r"[6-9]\d{9}\b"
    ),

    "IP_ADDRESS": (
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    )
}


def redact_with_regex(text: str) -> str:

    for pii_type, pattern in PII_PATTERNS.items():

        text = re.sub(
            pattern,
            f"[{pii_type}]",
            text
        )

    return text
```

Usage:

```python
text = """
Customer name is Rahul.
Email is rahul@example.com.
Phone is +91 9876543210.
Server IP is 192.168.1.10.
"""

cleaned = redact_with_regex(text)

print(cleaned)
```

Output:

```text
Customer name is Rahul.
Email is [EMAIL].
Phone is [PHONE].
Server IP is [IP_ADDRESS].
```

Notice:

```text
Rahul
```

was not removed.

Why?

Because regex is not good at detecting arbitrary names.

That is why regex alone is insufficient.

---

# 5. Detect PII using Named Entity Recognition

NER can detect contextual entities such as:

```text
PERSON
LOCATION
ORGANIZATION
DATE
```

For example, with spaCy:

```python
import spacy


nlp = spacy.load(
    "en_core_web_sm"
)


def redact_entities(text: str) -> str:

    doc = nlp(text)

    replacements = []

    for entity in doc.ents:

        if entity.label_ in {
            "PERSON",
            "GPE"
        }:

            replacements.append(
                (
                    entity.start_char,
                    entity.end_char,
                    f"[{entity.label_}]"
                )
            )

    for start, end, replacement in reversed(replacements):

        text = (
            text[:start]
            + replacement
            + text[end:]
        )

    return text
```

Example:

```python
text = (
    "John Smith lives in Bangalore "
    "and contacted customer support."
)

print(
    redact_entities(text)
)
```

Conceptually:

```text
[PERSON] lives in [GPE]
and contacted customer support.
```

However, generic NER models can:

```text
Miss names
Misclassify entities
Remove non-sensitive information
```

So again:

> **NER should be one layer in a multi-stage pipeline.**

---

# 6. Use Microsoft Presidio-style PII detection

For enterprise applications, a dedicated PII detection framework is often more appropriate.

Conceptually:

```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine


analyzer = AnalyzerEngine()

anonymizer = AnonymizerEngine()


text = """
My name is John Smith.
Email: john.smith@example.com
Phone: +1 555 123 4567
"""
```

Analyze:

```python
results = analyzer.analyze(
    text=text,

    entities=[
        "PERSON",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER"
    ],

    language="en"
)
```

Then anonymize:

```python
result = anonymizer.anonymize(
    text=text,
    analyzer_results=results
)

print(result.text)
```

Conceptually:

```text
My name is <PERSON>.
Email: <EMAIL_ADDRESS>
Phone: <PHONE_NUMBER>
```

This approach is useful because it combines multiple recognizers.

---

# 7. Replace instead of simply deleting

Suppose you delete the PII:

Original:

```text
Customer John Smith emailed john@example.com
about his payment.
```

Deleted:

```text
Customer emailed
about his payment.
```

The sentence quality is damaged.

Instead:

```text
Customer [PERSON] emailed [EMAIL]
about his payment.
```

This preserves:

```text
Grammar
Context
Semantic structure
Task behavior
```

For LLM training, placeholders are often better than blind deletion.

---

# 8. Pseudonymization

Sometimes you need consistency.

Suppose:

```text
John Smith submitted a request.
John Smith contacted support again.
```

You may want:

```text
PERSON_001 submitted a request.
PERSON_001 contacted support again.
```

This preserves relationships.

Example:

```python
import hashlib


def pseudonymize(
    value: str,
    entity_type: str
) -> str:

    digest = hashlib.sha256(
        value.encode()
    ).hexdigest()[:8]

    return (
        f"{entity_type}_{digest}"
    )
```

Example:

```python
pseudonymize(
    "John Smith",
    "PERSON"
)
```

Output:

```text
PERSON_a1b2c3d4
```

However, important production note:

> **Hashing is not automatically anonymization.**

For small or predictable identifier spaces, hashes can sometimes be brute-forced.

For strong privacy requirements, use:

```text
Tokenization
Encryption
Secure mapping service
Irreversible anonymization
```

depending on the use case.

---

# 9. Consistent pseudonymization with HMAC

A better approach for deterministic pseudonyms is a keyed HMAC:

```python
import hmac
import hashlib
import os


SECRET_KEY = os.environ["PII_TOKENIZATION_KEY"].encode()


def pseudonymize_value(
    value: str,
    entity_type: str
):

    digest = hmac.new(
        SECRET_KEY,
        value.strip().lower().encode(),
        hashlib.sha256
    ).hexdigest()[:12]

    return f"{entity_type}_{digest}"
```

Example:

```python
pseudonymize_value(
    "John Smith",
    "PERSON"
)
```

The same input consistently produces the same pseudonym while not exposing a plain unsalted hash.

In production, the secret should come from a secret manager, not source code.

---

# 10. Detect secrets separately from PII

Training data may contain:

```text
AWS keys
GitHub tokens
Database passwords
JWTs
API keys
Private keys
```

Example:

```text
AWS_SECRET_ACCESS_KEY=...
```

This may not technically be PII, but it must not enter the training set.

Create a separate secret scan:

```python
SECRET_PATTERNS = {
    "GENERIC_API_KEY": (
        r"(?i)(api[_-]?key)"
        r"\s*[:=]\s*['\"]?[\w\-]{16,}"
    )
}


def redact_secrets(text):

    for secret_type, pattern in (
        SECRET_PATTERNS.items()
    ):

        text = re.sub(
            pattern,
            f"{secret_type}=[REDACTED]",
            text
        )

    return text
```

For production, use dedicated secret-scanning tools and validators rather than relying only on simple regex.

---

# 11. Build a complete sanitization pipeline

A simplified architecture:

```python
class PIISanitizer:

    def __init__(self):

        self.pii_patterns = {
            "EMAIL": (
                r"\b[A-Za-z0-9._%+-]+"
                r"@[A-Za-z0-9.-]+"
                r"\.[A-Za-z]{2,}\b"
            ),

            "PHONE": (
                r"\b(?:\+91[-\s]?)?"
                r"[6-9]\d{9}\b"
            )
        }

    def redact_regex(
        self,
        text: str
    ) -> str:

        for pii_type, pattern in (
            self.pii_patterns.items()
        ):

            text = re.sub(
                pattern,
                f"[{pii_type}]",
                text
            )

        return text

    def redact_entities(
        self,
        text: str
    ) -> str:

        doc = nlp(text)

        entities = []

        for entity in doc.ents:

            if entity.label_ == "PERSON":

                entities.append(
                    (
                        entity.start_char,
                        entity.end_char,
                        "[PERSON]"
                    )
                )

        for start, end, replacement in reversed(entities):

            text = (
                text[:start]
                + replacement
                + text[end:]
            )

        return text

    def sanitize(
        self,
        text: str
    ) -> str:

        text = self.redact_regex(
            text
        )

        text = self.redact_entities(
            text
        )

        return text
```

Usage:

```python
sanitizer = PIISanitizer()

text = """
John Smith can be contacted at
john.smith@example.com or
+91 9876543210.
"""

cleaned = sanitizer.sanitize(
    text
)

print(cleaned)
```

Expected structure:

```text
[PERSON] can be contacted at
[EMAIL] or
[PHONE].
```

---

# 12. Apply sanitization to a Hugging Face dataset

Suppose your dataset contains:

```python
{
    "instruction": "Help the customer",
    "input": "My email is john@example.com",
    "output": "We will contact you"
}
```

Use `Dataset.map()`:

```python
from datasets import Dataset


def sanitize_example(example):

    return {

        "instruction":
            sanitizer.sanitize(
                example["instruction"]
            ),

        "input":
            sanitizer.sanitize(
                example["input"]
            ),

        "output":
            sanitizer.sanitize(
                example["output"]
            )
    }


clean_dataset = dataset.map(
    sanitize_example
)
```

Then inspect:

```python
print(
    clean_dataset[0]
)
```

This is important:

> **Scan both inputs and outputs.**

Don't assume that only the user input contains PII.

---

# 13. Add a reject policy

Some records should not be sanitized and retained.

Example:

```text
Original:

John Smith
Address: 123 Example Street
Aadhaar: XXXXXXXXXXXX
Bank account: XXXXX
```

After heavy sanitization:

```text
[PERSON]
[ADDRESS]
[GOVERNMENT_ID]
[BANK_ACCOUNT]
```

This may no longer provide useful training value.

So define policies:

```text
Low PII risk
    ↓
Sanitize and keep

Medium PII risk
    ↓
Human review

High PII risk
    ↓
Reject
```

Example:

```python
def should_reject(
    pii_count: int
) -> bool:

    MAX_ALLOWED_PII = 3

    return (
        pii_count
        > MAX_ALLOWED_PII
    )
```

In a real system, policy would depend on data classification, entity type, and legal/security requirements—not just a simple count.

---

# 14. Two-pass validation

A good production pattern is:

```text
PASS 1

Raw text
   │
   ▼
Detect PII
   │
   ▼
Redact


PASS 2

Redacted text
   │
   ▼
Detect PII again
   │
   ▼
Still detected?
   │
   ├── Yes → quarantine/reprocess
   │
   └── No → approved
```

Code:

```python
def validate_sanitization(text):

    remaining_pii = analyzer.analyze(
        text=text,

        language="en"
    )

    return len(
        remaining_pii
    ) == 0
```

Then:

```python
cleaned = sanitize(text)

if not validate_sanitization(cleaned):

    raise ValueError(
        "PII still detected"
    )
```

For real systems, I'd avoid throwing the raw text into logs when validation fails.

---

# 15. Keep audit metadata, not raw PII

Bad logging:

```python
logger.info(
    f"Found PII: {email}"
)
```

Now your logs contain PII.

Better:

```python
logger.info(
    "PII detected",
    extra={
        "entity_type": "EMAIL",
        "count": 1
    }
)
```

Track:

```text
Dataset version
Record ID
PII entity type
Number of detections
Sanitization status
Pipeline version
```

Avoid storing:

```text
Actual email
Actual phone number
Actual Aadhaar number
Actual account number
```

---

# 16. Use a structured audit record

For example:

```python
audit_record = {

    "record_id": "record_123",

    "dataset_version": "v4",

    "pipeline_version": "pii-v2",

    "entities_detected": {
        "EMAIL": 2,
        "PHONE": 1
    },

    "status": "sanitized"
}
```

This gives you observability without copying sensitive values into logs.

---

# 17. Validate data before it reaches training

The pipeline should be:

```text
              Raw Dataset
                   │
                   ▼
              PII Pipeline
                   │
          ┌────────┴────────┐
          │                 │
       Rejected          Sanitized
          │                 │
       Quarantine      Validation Scan
                            │
                            ▼
                     Approved Dataset
                            │
                            ▼
                        Training
```

I would **not** allow the trainer to directly read raw customer data.

Instead:

```text
Raw zone
   │
   │ restricted
   ▼
Sanitized zone
   │
   │ training access
   ▼
Training job
```

---

# 18. Detect PII quality metrics

Measure your pipeline.

For a labeled PII test dataset:

$$
Precision =
\frac{TP}{TP + FP}
$$

$$
Recall =
\frac{TP}{TP + FN}
$$

For PII removal, recall is especially important because:

```text
False Negative
    ↓
PII missed
    ↓
Sensitive data enters training
```

Example:

```python
def calculate_recall(
    true_positives,
    false_negatives
):

    return (
        true_positives
        /
        (
            true_positives
            +
            false_negatives
        )
    )
```

But you must also monitor precision:

```text
Too many false positives
       ↓
Useful training data destroyed
```

---

# 19. Production implementation design

A production pipeline might look like:

```text
                 Data Sources
                      │
          ┌───────────┼───────────┐
          │           │           │
         DBs        PDFs        APIs
          │           │           │
          └───────────┼───────────┘
                      │
                      ▼
                Ingestion Queue
                      │
                      ▼
              Data Classification
                      │
                      ▼
              PII Detection Layer
              ┌───────┼────────┐
              │       │        │
            Regex    NER    Custom Rules
              │       │        │
              └───────┼────────┘
                      │
                      ▼
                Sanitization
                      │
                      ▼
              Second PII Scan
                      │
                      ▼
              Quality Validation
                      │
                      ▼
              Versioned Dataset
                      │
                      ▼
               Fine-tuning Job
```

---

# 20. Important production considerations

## A. Don't send raw sensitive data to unnecessary services

If using an external PII detection API:

```text
Raw sensitive data
      ↓
Third-party API
```

may itself create a compliance issue.

For highly sensitive enterprise data, consider:

```text
On-premise detection
Private cloud deployment
VPC-contained services
Local models
```

depending on your security requirements.

---

## B. Minimize data collection

The best PII is:

> **PII you never collect in the first place.**

Before storing data:

```text
Do we need this field?
```

If not:

```text
Don't ingest it.
```

---

## C. Preserve original data separately

Sometimes you need raw data for operational purposes.

Use:

```text
Raw data
    ↓
Restricted storage

Sanitized data
    ↓
Training environment
```

Training should consume:

```text
Sanitized dataset only
```

---

# 21. A complete simplified implementation

```python
import re
from typing import Dict


class TrainingDataSanitizer:

    def __init__(self):

        self.patterns: Dict[str, str] = {

            "EMAIL": (
                r"\b[A-Za-z0-9._%+-]+"
                r"@[A-Za-z0-9.-]+"
                r"\.[A-Za-z]{2,}\b"
            ),

            "PHONE": (
                r"\b(?:\+\d{1,3}[-.\s]?)?"
                r"(?:\(?\d{2,4}\)?[-.\s]?)?"
                r"\d{6,10}\b"
            ),

            "IP_ADDRESS": (
                r"\b(?:\d{1,3}\.){3}"
                r"\d{1,3}\b"
            )
        }

    def sanitize(
        self,
        text: str
    ) -> str:

        for entity_type, pattern in (
            self.patterns.items()
        ):

            text = re.sub(
                pattern,
                f"[{entity_type}]",
                text
            )

        return text


sanitizer = TrainingDataSanitizer()


example = """
Customer John contacted us.

Email:
customer@example.com

Phone:
9876543210

Server:
10.0.0.15
"""


cleaned = sanitizer.sanitize(
    example
)


print(cleaned)
```

Then combine this with:

```text
NER
+
Dedicated PII recognizers
+
Custom organization-specific rules
+
Secret scanning
+
Second-pass validation
```

for production.

---

# 22. Interview answer

If an interviewer asks:

> **"How would you remove PII from training data?"**

You can answer:

> **I would implement a multi-stage data sanitization pipeline. First, I would classify the data and minimize sensitive fields at ingestion. For detection, I would combine regex and validation rules for structured identifiers such as emails and phone numbers with NER or a dedicated PII detection system for contextual information such as names and addresses. I would also run separate secret scanning for API keys and credentials.**
>
> **After detection, I would replace values with typed placeholders like `[EMAIL]` or `[PERSON]` when preserving the semantic structure is useful, and reject records that contain too much sensitive information to safely retain. For cases requiring consistent references, I could use controlled pseudonymization. I would then run a second PII scan on the sanitized output and quarantine failures.**
>
> **Finally, I would ensure training jobs only access the approved sanitized dataset, keep raw data in a restricted environment, version the sanitization pipeline, and log only aggregate detection metadata rather than the actual PII values.**

## Best mental model

```text
Don't trust one detector

Regex
  +
NER
  +
PII recognizer
  +
Custom rules
  +
Secret scanner
        ↓
    Sanitize
        ↓
  Scan again
        ↓
 Quarantine failures
        ↓
 Versioned safe dataset
        ↓
     Fine-tuning
```

That is the kind of **production-level answer** expected for a Senior AI/ML Engineer interview.
