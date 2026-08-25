# How would you fine-tune an LLM for SQL generation?

This is an excellent AI/LLM Engineer interview question.

The goal is to teach an LLM:

```text
Natural language question
        +
Database schema
        ↓
Correct SQL query
```

Example:

```text
User:
Show me all customers who spent more than 10,000.

Schema:
customers(id, name, total_spent)

        ↓

SQL:
SELECT id, name, total_spent
FROM customers
WHERE total_spent > 10000;
```

---

# 1. First understand what we are fine-tuning

We are **not simply training the model to write SQL syntax**.

Modern LLMs already know basic SQL.

We fine-tune when we need the model to reliably learn:

* company-specific schemas
* table relationships
* column names
* business terminology
* SQL style
* complex query patterns
* approved query patterns
* domain-specific metrics

For example:

```text
User:
Show active customers

Company definition of "active":
A customer who made at least one purchase in the last 90 days.
```

The correct SQL might be:

```sql
SELECT DISTINCT c.customer_id, c.name
FROM customers c
JOIN orders o
    ON c.customer_id = o.customer_id
WHERE o.created_at >= CURRENT_DATE - INTERVAL '90 days';
```

The difficult part is often **business semantics**, not SQL syntax.

---

# 2. Fine-tuning vs RAG for SQL generation

This distinction is extremely important.

## Fine-tuning

Use it for:

```text
How the model generates SQL
Domain patterns
SQL conventions
Business logic patterns
Output format
```

## RAG

Use it for:

```text
Current database schema
Table definitions
Column descriptions
Business glossary
Examples
```

A good production system is often:

```text
                 User Question
                       │
                       ▼
                 Schema Retrieval
                       │
                       ▼
               Business Glossary
                       │
                       ▼
                Fine-tuned LLM
                       │
                       ▼
                     SQL
                       │
                       ▼
                SQL Validation
                       │
              ┌────────┴────────┐
              ▼                 ▼
            Safe              Unsafe
              │                 │
              ▼                 ▼
           Execute            Reject
```

> You generally should not fine-tune a model every time your database schema changes.

---

# 3. Design the training dataset

A training example should contain:

```text
Database schema
        +
User question
        +
Correct SQL
```

Example:

```json
{
  "schema": {
    "customers": [
      "customer_id",
      "name",
      "email",
      "created_at"
    ],
    "orders": [
      "order_id",
      "customer_id",
      "amount",
      "status",
      "created_at"
    ]
  },
  "question": "Show customers who have placed more than 5 orders.",
  "sql": "SELECT c.customer_id, c.name FROM customers c JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.customer_id, c.name HAVING COUNT(o.order_id) > 5;"
}
```

For instruction tuning, convert it into chat format.

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert SQL generation assistant. Generate only valid PostgreSQL SQL. Never invent tables or columns."
    },
    {
      "role": "user",
      "content": "Database schema:\ncustomers(customer_id, name, email, created_at)\norders(order_id, customer_id, amount, status, created_at)\n\nQuestion: Show customers who have placed more than 5 orders."
    },
    {
      "role": "assistant",
      "content": "SELECT c.customer_id, c.name\nFROM customers c\nJOIN orders o ON c.customer_id = o.customer_id\nGROUP BY c.customer_id, c.name\nHAVING COUNT(o.order_id) > 5;"
    }
  ]
}
```

---

# 4. What types of examples should the dataset contain?

A good SQL-generation dataset should contain increasing difficulty.

## Level 1: Simple SELECT

```text
Show all customers.
```

```sql
SELECT *
FROM customers;
```

---

## Level 2: Filtering

```text
Show completed orders.
```

```sql
SELECT *
FROM orders
WHERE status = 'completed';
```

---

## Level 3: Aggregation

```text
Show total revenue.
```

```sql
SELECT SUM(amount)
FROM orders;
```

---

## Level 4: GROUP BY

```text
Show revenue by customer.
```

```sql
SELECT
    customer_id,
    SUM(amount) AS total_revenue
FROM orders
GROUP BY customer_id;
```

---

## Level 5: JOIN

```text
Show orders with customer names.
```

```sql
SELECT
    o.order_id,
    c.name
FROM orders o
JOIN customers c
    ON o.customer_id = c.customer_id;
```

---

## Level 6: Subqueries and CTEs

```text
Show customers whose spending is above average.
```

```sql
SELECT
    customer_id,
    total_spent
FROM customers
WHERE total_spent >
(
    SELECT AVG(total_spent)
    FROM customers
);
```

---

## Level 7: Window functions

```sql
SELECT
    customer_id,
    amount,
    RANK() OVER (
        PARTITION BY customer_id
        ORDER BY amount DESC
    ) AS order_rank
FROM orders;
```

Your training data should contain realistic distributions, not only easy examples.

---

# 5. Create the dataset with Python

Let's create a small dataset.

```python
import json
```

```python
examples = [
    {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert PostgreSQL assistant. "
                    "Generate only valid SQL. "
                    "Never invent tables or columns."
                )
            },
            {
                "role": "user",
                "content": """
Database schema:

customers(
    customer_id,
    name,
    email,
    created_at
)

Question:
Show all customers.
"""
            },
            {
                "role": "assistant",
                "content": """
SELECT *
FROM customers;
"""
            }
        ]
    },

    {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert PostgreSQL assistant. "
                    "Generate only valid SQL. "
                    "Never invent tables or columns."
                )
            },
            {
                "role": "user",
                "content": """
Database schema:

customers(
    customer_id,
    name
)

orders(
    order_id,
    customer_id,
    amount,
    status
)

Question:
Show customers who have spent more than 10000.
"""
            },
            {
                "role": "assistant",
                "content": """
SELECT
    c.customer_id,
    c.name,
    SUM(o.amount) AS total_spent
FROM customers c
JOIN orders o
    ON c.customer_id = o.customer_id
GROUP BY
    c.customer_id,
    c.name
HAVING SUM(o.amount) > 10000;
"""
            }
        ]
    }
]
```

Save as JSONL:

```python
with open(
    "sql_training.jsonl",
    "w",
    encoding="utf-8"
) as file:

    for example in examples:
        file.write(
            json.dumps(example)
            + "\n"
        )
```

---

# 6. Generate consistent training examples from a schema

In a real project, you might have metadata like this:

```python
database_schema = {
    "customers": [
        "customer_id",
        "name",
        "email",
        "created_at"
    ],

    "orders": [
        "order_id",
        "customer_id",
        "amount",
        "status",
        "created_at"
    ]
}
```

Convert schema to text:

```python
def format_schema(schema: dict) -> str:

    lines = []

    for table, columns in schema.items():

        columns_text = ", ".join(columns)

        lines.append(
            f"{table}({columns_text})"
        )

    return "\n".join(lines)
```

Usage:

```python
schema_text = format_schema(
    database_schema
)

print(schema_text)
```

Output:

```text
customers(customer_id, name, email, created_at)
orders(order_id, customer_id, amount, status, created_at)
```

Create an example:

```python
def create_training_example(
    schema: dict,
    question: str,
    sql: str
):

    schema_text = format_schema(schema)

    return {
        "messages": [

            {
                "role": "system",
                "content": (
                    "You are an expert PostgreSQL SQL generator. "
                    "Generate only executable SQL. "
                    "Use only tables and columns from the schema."
                )
            },

            {
                "role": "user",
                "content": f"""
Schema:

{schema_text}

Question:

{question}
"""
            },

            {
                "role": "assistant",
                "content": sql
            }
        ]
    }
```

Usage:

```python
example = create_training_example(

    schema=database_schema,

    question=(
        "Show customers with more "
        "than 5 orders."
    ),

    sql="""
SELECT
    c.customer_id,
    c.name
FROM customers c
JOIN orders o
    ON c.customer_id = o.customer_id
GROUP BY
    c.customer_id,
    c.name
HAVING COUNT(o.order_id) > 5;
"""
)
```

---

# 7. Clean the SQL dataset

This is extremely important.

A bad SQL dataset:

```text
Question:
Show all users

SQL:
SELECT * FROM user;
```

But your schema is:

```text
customers(customer_id, name)
```

The model will learn hallucination.

We should validate:

```text
Tables
Columns
SQL syntax
Business logic
```

---

# 8. Validate SQL syntax

A simple library can parse SQL.

For example:

```bash
pip install sqlglot
```

Python:

```python
import sqlglot
```

Validation:

```python
def validate_sql(sql: str) -> bool:

    try:

        sqlglot.parse_one(
            sql,
            dialect="postgres"
        )

        return True

    except Exception:

        return False
```

Usage:

```python
sql = """
SELECT *
FROM customers
"""

print(
    validate_sql(sql)
)
```

Output:

```text
True
```

Invalid:

```python
bad_sql = """
SELECT FROM customers
"""

print(
    validate_sql(bad_sql)
)
```

Output:

```text
False
```

---

# 9. Validate dangerous SQL

You usually should not train or execute unrestricted SQL.

Example:

```text
DROP TABLE customers;
DELETE FROM customers;
UPDATE customers;
```

For a read-only analytics assistant:

```python
FORBIDDEN_KEYWORDS = {
    "DROP",
    "DELETE",
    "UPDATE",
    "INSERT",
    "ALTER",
    "TRUNCATE",
    "GRANT",
    "REVOKE"
}
```

Validation:

```python
def is_safe_sql(sql: str) -> bool:

    upper_sql = sql.upper()

    return not any(
        keyword in upper_sql
        for keyword in FORBIDDEN_KEYWORDS
    )
```

Example:

```python
sql = """
SELECT *
FROM customers;
"""

print(
    is_safe_sql(sql)
)
```

For production, don't rely only on keyword matching because SQL can contain comments, nested expressions, multiple statements, etc. Use AST parsing plus database permissions.

---

# 10. Validate tables against schema

Suppose schema contains:

```python
ALLOWED_TABLES = {
    "customers",
    "orders"
}
```

Use `sqlglot`:

```python
import sqlglot
from sqlglot import exp
```

```python
def validate_tables(
    sql: str
):

    tree = sqlglot.parse_one(
        sql,
        dialect="postgres"
    )

    tables = {

        table.name

        for table in tree.find_all(
            exp.Table
        )
    }

    invalid_tables = (
        tables - ALLOWED_TABLES
    )

    if invalid_tables:

        raise ValueError(
            f"Invalid tables: {invalid_tables}"
        )

    return True
```

Example:

```python
sql = """
SELECT *
FROM customers
"""
```

```python
validate_tables(sql)
```

But:

```sql
SELECT *
FROM secret_table;
```

Produces:

```text
Invalid tables:
{'secret_table'}
```

---

# 11. Split the dataset correctly

This is important for text-to-SQL.

Do not randomly split examples if the same schema/question pattern appears in both training and validation.

Bad:

```text
TRAIN:
Show customers with spending > 1000

VALIDATION:
Show customers with spending > 2000
```

The validation is too similar.

Better:

```text
TRAIN:
Simple filters
Basic joins
Revenue queries

VALIDATION:
New schemas
New join combinations
New business questions
```

Example:

```python
from datasets import Dataset
```

```python
dataset = Dataset.from_list(
    examples
)
```

```python
split = dataset.train_test_split(

    test_size=0.2,

    seed=42
)
```

For production-quality evaluation, I would additionally create:

```text
In-domain test
Cross-schema test
Hard-query test
Adversarial test
```

---

# 12. Load the model

```python
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
```

```python
MODEL_NAME = (
    "your-instruct-model"
)
```

Tokenizer:

```python
tokenizer = (
    AutoTokenizer
    .from_pretrained(
        MODEL_NAME
    )
)
```

Model:

```python
model = (
    AutoModelForCausalLM
    .from_pretrained(
        MODEL_NAME
    )
)
```

---

# 13. Convert conversations using the chat template

```python
def format_example(example):

    text = tokenizer.apply_chat_template(

        example["messages"],

        tokenize=False,

        add_generation_prompt=False
    )

    return {
        "text": text
    }
```

Apply:

```python
dataset = dataset.map(
    format_example
)
```

The model sees:

```text
SYSTEM:
You are an expert PostgreSQL assistant.

USER:
Schema:
customers(...)
orders(...)

Question:
Show customers who spent more than 10000.

ASSISTANT:
SELECT ...
```

During training, the model learns to predict the SQL continuation.

---

# 14. Fine-tune with LoRA

```bash
pip install transformers peft trl datasets accelerate
```

Import:

```python
from peft import (
    LoraConfig,
    get_peft_model
)
```

Configuration:

```python
lora_config = LoraConfig(

    r=16,

    lora_alpha=32,

    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],

    task_type="CAUSAL_LM"
)
```

Apply:

```python
model = get_peft_model(
    model,
    lora_config
)
```

Check:

```python
model.print_trainable_parameters()
```

---

# 15. Train with supervised fine-tuning

```python
from trl import (
    SFTTrainer,
    SFTConfig
)
```

Configuration:

```python
training_args = SFTConfig(

    output_dir="./sql_generator",

    num_train_epochs=3,

    learning_rate=2e-4,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=8,

    max_length=1024,

    warmup_ratio=0.05,

    lr_scheduler_type="cosine",

    logging_steps=10,

    eval_strategy="steps",

    eval_steps=100,

    bf16=True,

    report_to="none"
)
```

Trainer:

```python
trainer = SFTTrainer(

    model=model,

    args=training_args,

    train_dataset=dataset["train"],

    processing_class=tokenizer,

    dataset_text_field="text"
)
```

Train:

```python
trainer.train()
```

Save:

```python
trainer.save_model(
    "./sql_generator_lora"
)
```

---

# 16. Better training: train mainly on the assistant response

This is a very important production detail.

During normal causal language-model training:

```text
SYSTEM TOKENS   → Loss
USER TOKENS     → Loss
ASSISTANT SQL → Loss
```

But we usually want:

```text
SYSTEM TOKENS   → ignored
USER TOKENS     → ignored
ASSISTANT SQL → calculate loss
```

Conceptually:

```text
Input:

[System]
You are a SQL generator.

[User]
Show all customers.

[Assistant]
SELECT * FROM customers;

Labels:

-100
-100
-100
...
SELECT
*
FROM
customers
;
```

`-100` means:

```text
Do not calculate training loss
for these tokens.
```

This focuses training on generating the correct SQL.

For chat fine-tuning, use the tokenizer/model's supported assistant-only loss mechanism when available rather than assuming every model uses the same masking format.

---

# 17. Inference

After fine-tuning:

```python
messages = [

    {
        "role": "system",

        "content": (
            "Generate valid PostgreSQL SQL. "
            "Use only the provided schema. "
            "Return SQL only."
        )
    },

    {
        "role": "user",

        "content": """
Schema:

customers(
    customer_id,
    name,
    email
)

orders(
    order_id,
    customer_id,
    amount
)

Question:

Show customers who spent
more than 10000.
"""
    }
]
```

Tokenize:

```python
inputs = tokenizer.apply_chat_template(

    messages,

    add_generation_prompt=True,

    return_tensors="pt"
).to(model.device)
```

Generate:

```python
output = model.generate(

    inputs,

    max_new_tokens=256,

    do_sample=False,

    temperature=None
)
```

Decode only the newly generated tokens:

```python
generated_tokens = output[
    0,
    inputs.shape[1]:
]

sql = tokenizer.decode(

    generated_tokens,

    skip_special_tokens=True
)

print(sql)
```

Expected:

```sql
SELECT
    c.customer_id,
    c.name,
    SUM(o.amount) AS total_spent
FROM customers c
JOIN orders o
    ON c.customer_id = o.customer_id
GROUP BY
    c.customer_id,
    c.name
HAVING SUM(o.amount) > 10000;
```

---

# 18. Validate before execution

Never directly execute LLM-generated SQL like this:

```python
cursor.execute(llm_sql)
```

Instead:

```text
Generate SQL
     ↓
Parse SQL
     ↓
Check syntax
     ↓
Check statement type
     ↓
Check tables/columns
     ↓
Check permissions
     ↓
Apply row/time limits
     ↓
Execute using read-only DB user
```

A basic validator:

```python
import sqlglot
from sqlglot import exp
```

```python
def validate_generated_sql(
    sql: str,
    allowed_tables: set[str]
) -> str:

    # Parse
    statements = sqlglot.parse(
        sql,
        dialect="postgres"
    )

    if len(statements) != 1:

        raise ValueError(
            "Only one SQL statement allowed"
        )

    statement = statements[0]

    # Read-only query
    if not isinstance(
        statement,
        (
            exp.Select,
            exp.Union,
            exp.Subquery
        )
    ):

        raise ValueError(
            "Only read-only queries allowed"
        )

    # Tables
    tables = {

        table.name

        for table in statement.find_all(
            exp.Table
        )
    }

    invalid_tables = (
        tables - allowed_tables
    )

    if invalid_tables:

        raise ValueError(
            f"Unauthorized tables: "
            f"{invalid_tables}"
        )

    return statement.sql(
        dialect="postgres"
    )
```

Usage:

```python
safe_sql = validate_generated_sql(

    sql=sql,

    allowed_tables={
        "customers",
        "orders"
    }
)
```

---

# 19. Add execution protection

Even valid `SELECT` queries can be expensive.

For example:

```sql
SELECT *
FROM orders
CROSS JOIN customers;
```

This could be very expensive.

You should apply:

```text
Read-only database user
Query timeout
Maximum rows
Query cost limits
Rate limiting
Audit logging
```

Example conceptually:

```python
async def execute_query(
    connection,
    sql: str
):

    await connection.execute(
        "SET statement_timeout = '5s'"
    )

    result = await connection.execute(
        sql
    )

    return result.fetchmany(
        1000
    )
```

The exact implementation depends on the database driver.

---

# 20. Production architecture

A strong production text-to-SQL architecture:

```text
                      USER
                        │
                        ▼
              "Show top customers"
                        │
                        ▼
                Intent Detection
                        │
                        ▼
               Schema / RAG Retrieval
                        │
          ┌─────────────┴─────────────┐
          │                           │
          ▼                           ▼
      Table Schema              Business Glossary
          │                           │
          └─────────────┬─────────────┘
                        ▼
                 Fine-tuned LLM
                        │
                        ▼
                     SQL
                        │
                        ▼
                SQL AST Parser
                        │
                        ▼
                Security Validation
                        │
             ┌──────────┴───────────┐
             │                      │
             ▼                      ▼
           Valid                  Invalid
             │                      │
             ▼                      ▼
       Read-only DB              Reject
             │
             ▼
          Results
             │
             ▼
        LLM Explanation
```

---

# 21. Example FastAPI implementation

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
```

Create app:

```python
app = FastAPI()
```

Request:

```python
class SQLRequest(BaseModel):

    question: str
```

Schema:

```python
DATABASE_SCHEMA = {
    "customers": [
        "customer_id",
        "name",
        "email"
    ],

    "orders": [
        "order_id",
        "customer_id",
        "amount"
    ]
}
```

Endpoint:

```python
@app.post("/generate-sql")
async def generate_sql(
    request: SQLRequest
):

    prompt = f"""
You are a PostgreSQL SQL generator.

Rules:

1. Use only provided tables.
2. Use only provided columns.
3. Generate SELECT queries only.
4. Return SQL only.

Schema:

{format_schema(DATABASE_SCHEMA)}

Question:

{request.question}
"""

    # Generate using model
    sql = generate_sql_from_model(
        prompt
    )

    try:

        validated_sql = (
            validate_generated_sql(

                sql,

                allowed_tables=set(
                    DATABASE_SCHEMA.keys()
                )
            )
        )

    except ValueError as error:

        raise HTTPException(

            status_code=400,

            detail=str(error)
        )

    return {

        "sql":
            validated_sql
    }
```

---

# 22. How do you evaluate SQL generation?

Do not rely only on:

```text
Exact string match
```

These two queries are different strings:

```sql
SELECT name FROM customers;
```

```sql
SELECT customers.name FROM customers;
```

But they can mean the same thing.

Use multiple metrics.

## 1. Syntax validity

```text
Can the SQL parser parse it?
```

## 2. Execution accuracy

```text
Does it execute successfully?
```

## 3. Execution equivalence

```text
Does it return the expected result?
```

This is often one of the most useful metrics.

Example:

```python
def execution_accuracy(
    predicted_result,
    expected_result
):

    return (
        predicted_result
        ==
        expected_result
    )
```

## 4. Schema adherence

```text
Did the model use only allowed tables?
Did it use valid columns?
```

## 5. Business logic correctness

Example:

```text
Question:
Monthly active users
```

A syntactically valid query can still be wrong if it calculates "active" incorrectly.

This requires semantic/business evaluation.

---

# 23. Common mistakes

## Mistake 1: Fine-tuning with only SQL

Bad:

```text
SELECT * FROM customers;
SELECT * FROM orders;
```

The model needs the relationship:

```text
Question
+
Schema
↓
SQL
```

---

## Mistake 2: No schema

Bad:

```text
Question:
Show customer revenue

Target:
SELECT ...
```

The model may memorize tables rather than learn schema grounding.

---

## Mistake 3: Using fine-tuning for changing schemas

If your schema changes every week:

```text
Fine-tune
↓
Schema changes
↓
Fine-tune again
```

This is expensive.

Better:

```text
Current schema
↓
RAG / schema retrieval
↓
LLM
```

---

## Mistake 4: Executing generated SQL directly

Never assume:

```text
Valid SQL
=
Safe SQL
```

Use validation and least-privilege database access.

---

# Interview-ready answer

If asked:

> **How would you fine-tune an LLM for SQL generation?**

A strong answer is:

> **I would create a supervised fine-tuning dataset containing the natural-language question, the relevant database schema and business context, and the correct SQL query. The dataset should include different query complexities such as filtering, aggregation, joins, subqueries, CTEs, and window functions. I would clean and validate every SQL query and ensure it uses valid tables and columns.**
>
> **I would usually start with an instruction-tuned base model and use LoRA or QLoRA for cost-efficient supervised fine-tuning. During training, I would ideally calculate loss primarily on the assistant SQL output rather than the system and user instructions.**
>
> **In production, I would not depend only on fine-tuning. I would dynamically retrieve the current database schema and business glossary using RAG or metadata retrieval, generate SQL, parse it into an AST, validate the statement type and allowed tables and columns, and execute it using a read-only database user with timeouts and row limits.**
>
> **For evaluation, I would measure syntax validity, execution accuracy, schema adherence, and business-logic correctness rather than relying only on exact string matching.**

# Final mental model

```text
FINE-TUNING
    ↓
Teaches SQL generation behavior

RAG / METADATA RETRIEVAL
    ↓
Provides current schema and business knowledge

VALIDATION
    ↓
Ensures generated SQL is allowed

DATABASE PERMISSIONS
    ↓
Prevents unsafe execution
```

> **The best production text-to-SQL system is not just a fine-tuned model. It is Fine-tuning + Schema Retrieval + SQL Validation + Read-only Database Security.**
