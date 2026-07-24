PII (Personally Identifiable Information) protection is a critical requirement for production AI systems. If an LLM leaks personal data such as names, phone numbers, Aadhaar numbers, Social Security numbers, or credit card numbers, it can violate privacy regulations (such as GDPR, HIPAA, or PCI DSS depending on the data and jurisdiction) and expose organizations to legal and financial risk.

A secure AI application should **detect, protect, and control** PII throughout the request lifecycle.

---

# What is PII?

Examples of PII include:

```text
Name
Email Address
Phone Number
Passport Number
Aadhaar Number
PAN Number
Social Security Number
Credit Card Number
Date of Birth
Home Address
```

Example:

```text
John Doe

Email:
john@gmail.com

Phone:
9876543210

Credit Card:
4111-1111-1111-1111
```

---

# Where Can PII Leak?

```text
              User Input
                  │
                  ▼
            API Gateway
                  │
                  ▼
               LLM Agent
        ┌─────────┼─────────┐
        ▼         ▼         ▼
      Logs      RAG      External APIs
        │         │         │
        └─────────┼─────────┘
                  ▼
              User Output
```

PII may appear in:

* User prompts
* Conversation history
* Retrieved documents
* Tool calls
* Logs
* Model outputs

Every stage must be protected.

---

# Layer 1: Detect PII Before the LLM

Example:

```text
User:

My Aadhaar number is
1234-5678-9012

Can you summarize this?
```

Pipeline

```text
User

↓

PII Detector

↓

Mask Data

↓

LLM
```

---

## Simple Detection

```python
import re

EMAIL = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"

PHONE = r"\b\d{10}\b"

text = """
Email: john@gmail.com
Phone: 9876543210
"""

emails = re.findall(EMAIL, text)
phones = re.findall(PHONE, text)

print(emails)
print(phones)
```

Output

```text
['john@gmail.com']

['9876543210']
```

Regular expressions work for simple patterns, but they don't detect many real-world cases.

---

# Layer 2: Mask PII

Instead of

```text
John Doe

john@gmail.com

9876543210
```

Store

```text
[NAME]

[EMAIL]

[PHONE]
```

Python

```python
def mask(text):

    text = re.sub(
        EMAIL,
        "[EMAIL]",
        text
    )

    text = re.sub(
        PHONE,
        "[PHONE]",
        text
    )

    return text
```

---

# Layer 3: Named Entity Recognition (NER)

Regex cannot detect:

```text
John Smith

Microsoft

Paris
```

NER models can.

Example

```python
import spacy

nlp = spacy.load("en_core_web_sm")

doc = nlp(
    "John lives in London."
)

for ent in doc.ents:
    print(ent.text, ent.label_)
```

Output

```text
John PERSON

London GPE
```

---

# Layer 4: Microsoft Presidio

A production-grade open-source PII detection framework.

Architecture

```text
Text

↓

Presidio Analyzer

↓

PII Entities

↓

Presidio Anonymizer

↓

Masked Text
```

Example

```python
from presidio_analyzer import AnalyzerEngine

analyzer = AnalyzerEngine()

results = analyzer.analyze(
    text="Email me at john@gmail.com",
    language="en"
)
```

Output

```text
EMAIL_ADDRESS
```

Then anonymize it before sending it to the LLM.

---

# Layer 5: Protect Logs

Never log:

```text
User:
My card is

4111-1111-1111-1111
```

Instead

```text
User:
My card is

[CARD]
```

Bad

```python
logger.info(prompt)
```

Good

```python
logger.info(
    mask(prompt)
)
```

---

# Layer 6: Prevent Output Leakage

Suppose retrieved document contains

```text
Employee Salary

John

$200000
```

User asks

```text
Show all salaries.
```

Pipeline

```text
LLM

↓

Output Validator

↓

User
```

Python

```python
if contains_pii(answer):
    raise Exception(
        "Sensitive information detected"
    )
```

---

# Layer 7: Encrypt Stored Data

Never store conversations as plain text.

Bad

```text
Database

↓

Plain Text
```

Good

```text
Database

↓

AES-256 Encryption
```

Data should also be protected in transit with HTTPS/TLS.

---

# Layer 8: Tenant Isolation

Suppose:

```text
Company A

Company B
```

Users from Company A should never retrieve Company B's documents.

Correct retrieval:

```python
retriever.search(
    query,
    tenant_id=current_user.tenant_id
)
```

Even if the LLM is prompted to retrieve another tenant's data, the application should never return it.

---

# Layer 9: Minimize Data Collection

Instead of sending:

```text
Entire User Profile
```

Send only:

```text
First Name

Order Number
```

Follow the principle of **data minimization**: only process what is required for the task.

---

# Layer 10: Human Approval

High-risk operations

```text
Export Customer Database

↓

LLM

↓

Human Approval

↓

Execute
```

Never allow an autonomous agent to export sensitive information without appropriate controls.

---

# Production Pipeline

```text
                 User
                   │
                   ▼
          PII Detection
                   │
                   ▼
         Mask Sensitive Data
                   │
                   ▼
                LLM
                   │
                   ▼
        Output Validation
                   │
                   ▼
         Audit Logging
                   │
                   ▼
                 User
```

---

# Complete Example

```python
import re

EMAIL = r"\S+@\S+"

PHONE = r"\d{10}"

def sanitize(text):

    text = re.sub(
        EMAIL,
        "[EMAIL]",
        text
    )

    text = re.sub(
        PHONE,
        "[PHONE]",
        text
    )

    return text


user_prompt = """
Email:
john@gmail.com

Phone:
9876543210
"""

safe_prompt = sanitize(user_prompt)

response = llm.invoke(safe_prompt)
```

The LLM receives

```text
Email:
[EMAIL]

Phone:
[PHONE]
```

instead of the original values.

---

# Enterprise Architecture

```text
                    User
                      │
                      ▼
                API Gateway
                      │
                      ▼
             Authentication
                      │
                      ▼
              PII Detection
                      │
                      ▼
             Mask/Tokenize PII
                      │
                      ▼
                   Agent
          ┌──────────┼──────────┐
          ▼          ▼          ▼
         LLM        RAG       Tools
          │          │          │
          └──────────┼──────────┘
                     ▼
           Output PII Validation
                     │
                     ▼
             Encrypted Storage
                     │
                     ▼
               Audit Logs
```

---

# What Senior AI Teams Use

| Layer          | Common Technologies                                                            |
| -------------- | ------------------------------------------------------------------------------ |
| Detection      | Microsoft Presidio, spaCy NER, custom regex                                    |
| Masking        | Presidio Anonymizer, custom masking/tokenization                               |
| Encryption     | TLS for transport, AES-256 for storage                                         |
| Secrets        | AWS Secrets Manager, Azure Key Vault, HashiCorp Vault                          |
| Logging        | Structured logs with automatic PII redaction                                   |
| Access Control | RBAC/ABAC, tenant isolation                                                    |
| Monitoring     | Alerts for PII leakage and unusual access patterns                             |
| Governance     | Data retention policies, audit trails, human approval for sensitive operations |

## Interview Answer

If asked **"How do you protect PII in an AI application?"**, a strong answer is:

> "I protect PII throughout the entire pipeline. I detect sensitive information before it reaches the model using tools such as Microsoft Presidio, NER models, and regex where appropriate. I mask or tokenize sensitive values, encrypt data in transit and at rest, avoid logging raw PII, enforce tenant isolation and role-based access control, validate model outputs for accidental disclosure, and apply data minimization so the model only receives the information it needs. I also monitor for potential PII leakage and maintain audit logs for compliance."

This demonstrates that you understand PII protection as an **end-to-end system design concern**, not just a preprocessing step before calling an LLM.
