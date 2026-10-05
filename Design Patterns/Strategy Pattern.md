Absolutely. The **Strategy Pattern** is especially important for your AI projects because you have many places where the system may need to choose between different implementations at runtime—LLM providers, retrieval strategies, rerankers, risk calculations, routing policies, and so on.

The easiest definition is:

> **Strategy Pattern means defining multiple interchangeable ways of doing the same job, putting each way in its own class, and letting another component choose which strategy to use.**

---

# 1. Start with the problem

Imagine your financial AI system needs to retrieve documents.

You could retrieve documents using:

```text
Keyword search
Vector search
Hybrid search
Semantic search
```

A naive implementation might be:

```python
class Retriever:

    def retrieve(self, query, strategy):

        if strategy == "keyword":
            # keyword search
            ...

        elif strategy == "vector":
            # vector search
            ...

        elif strategy == "hybrid":
            # hybrid search
            ...
```

This works initially.

But as the application grows:

```python
if strategy == "keyword":
    ...

elif strategy == "vector":
    ...

elif strategy == "hybrid":
    ...

elif strategy == "bm25":
    ...

elif strategy == "reranked":
    ...

elif strategy == "multi_query":
    ...

elif strategy == "parent_child":
    ...
```

Your class becomes huge.

This is exactly the kind of problem the **Strategy Pattern** solves.

---

# 2. The Strategy Pattern idea

Instead of:

```text
Retriever
   |
   ├── if keyword
   ├── if vector
   ├── if hybrid
   └── if bm25
```

we create:

```text
                  RetrievalStrategy
                         ▲
            ┌────────────┼────────────┐
            │            │            │
            │            │            │
         Keyword       Vector       Hybrid
         Strategy      Strategy      Strategy
```

Then:

```text
                 RetrieverService
                       │
                       ▼
              RetrievalStrategy
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Keyword       Vector       Hybrid
```

The service doesn't care **how** retrieval happens.

It only knows:

```python
strategy.retrieve(query)
```

---

# 3. Simple Python example

Let's start with something unrelated to AI.

Suppose we have payment processing.

We support:

```text
Credit Card
PayPal
UPI
```

Without Strategy Pattern:

```python
class PaymentService:

    def pay(self, method, amount):

        if method == "card":
            print(f"Paying ₹{amount} using card")

        elif method == "paypal":
            print(f"Paying ₹{amount} using PayPal")

        elif method == "upi":
            print(f"Paying ₹{amount} using UPI")
```

Usage:

```python
service = PaymentService()

service.pay("card", 1000)
service.pay("upi", 500)
```

The problem is that `PaymentService` knows about every payment implementation.

---

# 4. Create the Strategy interface

With Strategy Pattern:

```python
from abc import ABC, abstractmethod


class PaymentStrategy(ABC):

    @abstractmethod
    def pay(self, amount: float) -> None:
        pass
```

This defines the contract.

Every payment strategy must implement:

```python
pay(amount)
```

---

# 5. Create concrete strategies

### Credit Card

```python
class CreditCardStrategy(PaymentStrategy):

    def pay(self, amount: float) -> None:

        print(
            f"Paying ₹{amount} using Credit Card"
        )
```

### PayPal

```python
class PayPalStrategy(PaymentStrategy):

    def pay(self, amount: float) -> None:

        print(
            f"Paying ₹{amount} using PayPal"
        )
```

### UPI

```python
class UPIStrategy(PaymentStrategy):

    def pay(self, amount: float) -> None:

        print(
            f"Paying ₹{amount} using UPI"
        )
```

Now we have:

```text
PaymentStrategy
      ▲
      │
 ┌────┼─────────────┐
 │    │             │
 ▼    ▼             ▼
Card PayPal        UPI
```

---

# 6. Create the Context

The **Context** is the object that uses the strategy.

```python
class PaymentService:

    def __init__(
        self,
        strategy: PaymentStrategy
    ):
        self.strategy = strategy

    def pay(self, amount: float):

        self.strategy.pay(amount)
```

Notice something important.

`PaymentService` doesn't know:

```text
CreditCard
PayPal
UPI
```

It only knows:

```text
PaymentStrategy
```

---

# 7. Using the Strategy Pattern

```python
card = CreditCardStrategy()

service = PaymentService(card)

service.pay(1000)
```

Output:

```text
Paying ₹1000 using Credit Card
```

Switch strategy:

```python
upi = UPIStrategy()

service = PaymentService(upi)

service.pay(1000)
```

Output:

```text
Paying ₹1000 using UPI
```

The `PaymentService` didn't change.

Only the strategy changed.

---

# 8. This is the key idea

The following:

```python
service = PaymentService(
    CreditCardStrategy()
)
```

means:

> Use credit card behavior.

Whereas:

```python
service = PaymentService(
    UPIStrategy()
)
```

means:

> Use UPI behavior.

So:

```text
Same Context
     │
     ├── Strategy A
     │
     ├── Strategy B
     │
     └── Strategy C
```

The behavior can be changed without modifying the Context.

---

# 9. Strategy Pattern vs simple function

You might ask:

> "Why not just use functions?"

You absolutely can in Python.

For example:

```python
def pay_by_card(amount):
    print("Card payment")


def pay_by_upi(amount):
    print("UPI payment")
```

And:

```python
class PaymentService:

    def __init__(self, payment_function):
        self.payment_function = payment_function

    def pay(self, amount):
        self.payment_function(amount)
```

This is conceptually still the same idea.

The Strategy Pattern is fundamentally about:

> **Encapsulating interchangeable behavior behind a common contract.**

In enterprise systems, classes/interfaces often make the contract, dependency injection, testing, and extension points clearer.

---

# 10. Now apply Strategy Pattern to your AI project

This is where it becomes much more relevant to you.

Suppose your RAG system supports:

```text
Vector Search
BM25
Hybrid Search
```

We create:

```python
from abc import ABC, abstractmethod


class RetrievalStrategy(ABC):

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int
    ):
        pass
```

This is our Strategy interface.

---

# 11. Vector strategy

```python
class VectorRetrievalStrategy(
    RetrievalStrategy
):

    def __init__(self, vector_store):
        self.vector_store = vector_store

    async def retrieve(
        self,
        query: str,
        top_k: int
    ):

        embedding = await self.vector_store.embed(query)

        return await self.vector_store.similarity_search(
            embedding=embedding,
            top_k=top_k
        )
```

---

# 12. BM25 strategy

```python
class BM25RetrievalStrategy(
    RetrievalStrategy
):

    def __init__(self, search_engine):
        self.search_engine = search_engine

    async def retrieve(
        self,
        query: str,
        top_k: int
    ):

        return await self.search_engine.bm25_search(
            query=query,
            top_k=top_k
        )
```

---

# 13. Hybrid strategy

```python
class HybridRetrievalStrategy(
    RetrievalStrategy
):

    def __init__(
        self,
        vector_store,
        search_engine
    ):
        self.vector_store = vector_store
        self.search_engine = search_engine

    async def retrieve(
        self,
        query: str,
        top_k: int
    ):

        vector_results = await (
            self.vector_store.semantic_search(
                query,
                top_k
            )
        )

        keyword_results = await (
            self.search_engine.bm25_search(
                query,
                top_k
            )
        )

        return self.merge(
            vector_results,
            keyword_results
        )

    def merge(
        self,
        vector_results,
        keyword_results
    ):

        # Production implementation could use
        # Reciprocal Rank Fusion or another
        # deterministic fusion algorithm.

        return vector_results + keyword_results
```

Now:

```text
             RetrievalStrategy
                    ▲
                    │
       ┌────────────┼─────────────┐
       │            │             │
       ▼            ▼             ▼
    Vector         BM25         Hybrid
   Strategy      Strategy      Strategy
```

---

# 14. The Context

Now create:

```python
class RetrievalService:

    def __init__(
        self,
        strategy: RetrievalStrategy
    ):
        self.strategy = strategy

    async def retrieve(
        self,
        query: str,
        top_k: int = 10
    ):

        return await self.strategy.retrieve(
            query,
            top_k
        )
```

This is the Context.

It doesn't know:

```text
pgvector
Qdrant
BM25
OpenSearch
Elasticsearch
```

It only knows:

```text
RetrievalStrategy
```

---

# 15. Runtime strategy selection

Now suppose your configuration says:

```python
retrieval_mode = "hybrid"
```

You can construct:

```python
strategy = HybridRetrievalStrategy(
    vector_store=vector_store,
    search_engine=search_engine
)

service = RetrievalService(strategy)

results = await service.retrieve(
    "What are the rules for guaranteed returns?"
)
```

For another request:

```python
strategy = VectorRetrievalStrategy(
    vector_store
)

service = RetrievalService(strategy)
```

Same service.

Different behavior.

---

# 16. Strategy Pattern in your financial compliance project

Your compliance system can use Strategy Pattern for several things.

For example:

```text
ComplianceReviewService
          │
          ▼
   RiskAssessmentStrategy
          │
    ┌─────┼──────┐
    ▼     ▼      ▼
RuleBased LLM    Hybrid
Risk      Risk   Risk
```

### Interface

```python
class RiskAssessmentStrategy(ABC):

    @abstractmethod
    async def assess(
        self,
        material: str
    ):
        pass
```

### Rule-based strategy

```python
class RuleBasedRiskStrategy(
    RiskAssessmentStrategy
):

    async def assess(
        self,
        material: str
    ):

        text = material.lower()

        if "guaranteed" in text:
            return "HIGH"

        if "risk-free" in text:
            return "HIGH"

        return "LOW"
```

### LLM strategy

```python
class LLMRiskStrategy(
    RiskAssessmentStrategy
):

    def __init__(self, llm):
        self.llm = llm

    async def assess(
        self,
        material: str
    ):

        prompt = f"""
        Assess the compliance risk of:

        {material}

        Return:
        LOW, MEDIUM or HIGH
        """

        return await self.llm.generate(prompt)
```

### Hybrid strategy

```python
class HybridRiskStrategy(
    RiskAssessmentStrategy
):

    def __init__(
        self,
        rule_engine,
        llm
    ):
        self.rule_engine = rule_engine
        self.llm = llm

    async def assess(
        self,
        material: str
    ):

        deterministic_result = (
            self.rule_engine.assess(material)
        )

        if deterministic_result == "HIGH":
            return "HIGH"

        return await self.llm.generate(
            f"Assess this material: {material}"
        )
```

Then:

```python
class ComplianceReviewService:

    def __init__(
        self,
        risk_strategy: RiskAssessmentStrategy
    ):
        self.risk_strategy = risk_strategy

    async def review(
        self,
        material: str
    ):

        risk = await self.risk_strategy.assess(
            material
        )

        return {
            "material": material,
            "risk": risk
        }
```

This is a very good enterprise use of Strategy.

---

# 17. Another example: LLM routing

Your AI platform might support:

```text
GPT
Claude
Llama
Bedrock
```

You could define:

```python
class LLMStrategy(ABC):

    @abstractmethod
    async def generate(
        self,
        prompt: str
    ):
        pass
```

Then:

```text
              LLMStrategy
                   ▲
        ┌──────────┼──────────┐
        ▼          ▼          ▼
      OpenAI     Claude      Llama
     Strategy   Strategy    Strategy
```

Your service:

```python
class AIService:

    def __init__(
        self,
        strategy: LLMStrategy
    ):
        self.strategy = strategy

    async def answer(
        self,
        prompt: str
    ):

        return await self.strategy.generate(
            prompt
        )
```

Now you can implement routing:

```python
class LLMRouter:

    def select(
        self,
        request_type: str
    ) -> LLMStrategy:

        if request_type == "complex":
            return OpenAIStrategy()

        if request_type == "cheap":
            return LlamaStrategy()

        return ClaudeStrategy()
```

Then:

```python
strategy = router.select("complex")

service = AIService(strategy)

answer = await service.answer(
    "Analyze this portfolio..."
)
```

---

# 18. Strategy vs Adapter

This is **very important for the architecture you're learning** because Strategy and Adapter can look almost identical.

### Adapter

The purpose is:

> **Make an incompatible external API conform to the interface my application expects.**

Example:

```text
My application expects:

LLMPort
   │
   ▼
OpenAIAdapter
   │
   ▼
OpenAI SDK
```

Adapter solves **interface incompatibility**.

---

### Strategy

The purpose is:

> **Choose between different algorithms/behaviors for accomplishing the same task.**

Example:

```text
RetrievalStrategy
       │
 ┌─────┼──────┐
 ▼     ▼      ▼
BM25 Vector Hybrid
```

Strategy solves **interchangeable behavior**.

---

# 19. But they can be used together

This is extremely common in your enterprise AI architecture.

For example:

```text
                 RAG Service
                     │
                     ▼
              RetrievalStrategy
                     │
              ┌──────┴──────┐
              ▼             ▼
        VectorStrategy   HybridStrategy
              │             │
              ▼             ▼
       VectorStorePort  VectorStorePort
              │             │
              ▼             ▼
        QdrantAdapter   PgVectorAdapter
```

Here:

### Strategy

```text
VectorStrategy
HybridStrategy
```

chooses **retrieval behavior**.

### Adapter

```text
QdrantAdapter
PgVectorAdapter
```

connects your application to **external infrastructure**.

So:

```text
Strategy = HOW should I perform the job?

Adapter = HOW do I connect my application to this external system?
```

That's a very useful distinction.

---

# 20. Strategy vs Factory

Another common interview question.

### Strategy

Defines the behavior:

```text
RetrievalStrategy
     ├── Vector
     ├── BM25
     └── Hybrid
```

### Factory

Creates the appropriate object:

```python
class RetrievalStrategyFactory:

    @staticmethod
    def create(
        strategy_name: str,
        dependencies
    ):

        if strategy_name == "vector":
            return VectorStrategy(
                dependencies.vector_store
            )

        if strategy_name == "bm25":
            return BM25Strategy(
                dependencies.search_engine
            )

        if strategy_name == "hybrid":
            return HybridStrategy(
                dependencies.vector_store,
                dependencies.search_engine
            )

        raise ValueError(
            f"Unknown strategy: {strategy_name}"
        )
```

Then:

```python
strategy = RetrievalStrategyFactory.create(
    "hybrid",
    dependencies
)

service = RetrievalService(strategy)
```

So:

```text
Factory
   ↓
creates Strategy

Strategy
   ↓
performs behavior
```

---

# 21. Strategy vs Template Method

Another distinction:

### Strategy

Behavior is selected by **composition**.

```python
service = RetrievalService(
    HybridRetrievalStrategy(...)
)
```

### Template Method

Behavior is customized through **inheritance**.

```python
class RAGPipeline:

    async def run(self, query):

        docs = await self.retrieve(query)

        context = self.build_context(docs)

        return await self.generate(context)

    async def retrieve(self, query):
        raise NotImplementedError
```

Then subclasses override:

```python
class HybridRAGPipeline(RAGPipeline):

    async def retrieve(self, query):
        ...
```

Strategy generally gives you more runtime flexibility because you can inject another strategy without creating a new Context subclass.

---

# 22. When should you use Strategy?

Use it when you have:

### 1. Multiple algorithms

```text
BM25
Vector
Hybrid
```

### 2. Multiple providers

```text
OpenAI
Claude
Bedrock
Llama
```

### 3. Multiple policies

```text
Strict
Balanced
Permissive
```

### 4. Multiple pricing models

```text
Standard
Premium
Enterprise
```

### 5. Multiple risk calculations

```text
Rule-based
ML-based
LLM-based
Hybrid
```

### 6. Multiple ranking approaches

```text
CrossEncoder
LLM reranker
Reciprocal Rank Fusion
Weighted ranking
```

---

# 23. When NOT to use Strategy

Don't do this:

```text
Strategy
   └── AddTwoNumbersStrategy
```

when there's only one algorithm.

If you have:

```python
def calculate_total(price, tax):
    return price + tax
```

you don't need:

```text
CalculationStrategy
CalculationContext
CalculationFactory
CalculationAdapter
CalculationPort
```

That would be overengineering.

Use Strategy when there is **real variability**.

---

# 24. The complete enterprise AI picture

In your modular AI platform, you can have:

```text
                     APPLICATION
                          │
        ┌─────────────────┼──────────────────┐
        │                 │                  │
        ▼                 ▼                  ▼
 RetrievalStrategy   LLMStrategy      RiskStrategy
        │                 │                  │
   ┌────┼────┐       ┌────┼────┐       ┌────┼────┐
   ▼    ▼    ▼       ▼    ▼    ▼       ▼    ▼    ▼
 BM25 Vector Hybrid OpenAI Claude Llama Rule LLM Hybrid
```

Then adapters connect those strategies to infrastructure:

```text
Strategy
    │
    ▼
Port
    │
    ▼
Adapter
    │
    ▼
Infrastructure
```

For example:

```text
HybridRetrievalStrategy
          │
          ▼
    VectorStorePort
          │
          ▼
    PgVectorAdapter
          │
          ▼
      PostgreSQL
```

This is how Strategy, Ports/Adapters, and Dependency Injection can work together rather than being competing patterns.

---

# 25. The interview answer you should remember

If someone asks:

> **"What is the Strategy Pattern?"**

You can answer:

> "Strategy Pattern encapsulates a family of interchangeable algorithms behind a common interface. The Context depends on that abstraction rather than concrete implementations, allowing the algorithm to be selected or changed without modifying the Context. In our AI platform, for example, we used a RetrievalStrategy abstraction with BM25, vector and hybrid implementations. The RAG service depends only on the strategy interface, so retrieval behavior can change without modifying the RAG orchestration logic."

Then explain the difference:

> "I use Strategy when I need interchangeable behavior, whereas I use Adapter when I need to make an external implementation conform to an interface expected by my application."

That distinction is particularly important when explaining your **Hexagonal + Strategy + Repository + Planner/Supervisor** architecture.
