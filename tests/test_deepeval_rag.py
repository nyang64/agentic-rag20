"""
DeepEval test suite for the Agentic-RAG pipeline.

Covers:
  - AnswerRelevancyMetric   – does the answer address the question?
  - FaithfulnessMetric      – is the answer grounded in the retrieved context?
  - ContextualPrecisionMetric – are the relevant contexts ranked at the top?
  - ContextualRecallMetric  – does the retrieved context cover the expected answer?
  - ContextualRelevancyMetric – are retrieved chunks relevant to the question?

Test modes
----------
Static (no real API or DB needed):
  Uses pre-built QA pairs from conftest.SAMPLE_QA.
  Requires OPENROUTER_API_KEY for the LLM judge.

Integration (marked @pytest.mark.integration):
  Calls the actual RAG pipeline and live retriever.
  Requires PGVECTOR_DB_URL and OPENROUTER_API_KEY.
  Run with:  pytest tests/test_deepeval_rag.py -m integration
"""

import os
import pytest
from dotenv import load_dotenv

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM

load_dotenv()

from tests.conftest import SAMPLE_QA, BAD_ANSWER_CASE


# ---------------------------------------------------------------------------
# Custom evaluator LLM that routes through OpenRouter
# ---------------------------------------------------------------------------

class OpenRouterEvalLLM(DeepEvalBaseLLM):
    """Wraps an OpenAI-compatible client (OpenRouter) for use as a DeepEval judge.

    Uses instructor for structured output so that free/OSS models (which often
    truncate or add stray characters in raw JSON mode) get automatic retry and
    prompt-level coercion instead of relying on native function-calling.
    """

    def __init__(self):
        import instructor
        from openai import OpenAI

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")

        self._model = os.getenv("OPENAI_FREE_MODEL", "openai/gpt-oss-20b:free")
        raw_client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        self._client = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)
        self._raw_client = raw_client

    def load_model(self):
        return self._client

    def generate(self, prompt: str, schema=None):
        if schema is not None:
            return self._client.chat.completions.create(
                model=self._model,
                response_model=schema,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                max_retries=3,
            )
        response = self._raw_client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return self._model


# ---------------------------------------------------------------------------
# Fixture: evaluator LLM (session-scoped to avoid repeated cold starts)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def eval_llm():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set – skipping DeepEval LLM-judge tests")
    return OpenRouterEvalLLM()


# ---------------------------------------------------------------------------
# Helper: build an LLMTestCase from a SAMPLE_QA dict
# ---------------------------------------------------------------------------

def _make_test_case(sample: dict) -> LLMTestCase:
    return LLMTestCase(
        input=sample["question"],
        actual_output=sample["answer"],
        expected_output=sample["ground_truth"],
        retrieval_context=sample["contexts"],
    )


# ===========================================================================
# 1. Answer Relevancy
#    Checks that the answer is on-topic and directly addresses the question.
# ===========================================================================

class TestAnswerRelevancy:
    def test_relevant_answers(self, eval_llm):
        """Representative samples should score ≥ 0.5 on answer relevancy."""
        metric = AnswerRelevancyMetric(threshold=0.5, model=eval_llm, async_mode=False)
        failures = []
        for qa in SAMPLE_QA[:2]:  # pgvector + retrieval — most stable samples
            case = _make_test_case(qa)
            metric.measure(case)
            if metric.score < 0.5:
                failures.append(f"{qa['question'][:50]}: {metric.score:.2f}")
        assert not failures, f"Low answer-relevancy scores: {failures}"

    def test_bad_answer_irrelevant(self, eval_llm):
        """MySQL/Redis answer to a pgvector question should score below 0.5."""
        case = _make_test_case(BAD_ANSWER_CASE)
        metric = AnswerRelevancyMetric(threshold=0.5, model=eval_llm, async_mode=False)
        metric.measure(case)
        assert metric.score < 0.5, (
            f"Expected low relevancy for bad answer, got {metric.score:.2f}"
        )


# ===========================================================================
# 2. Faithfulness
#    Checks that every claim in the answer is supported by retrieved context.
# ===========================================================================

class TestFaithfulness:
    def test_faithful_answers(self, eval_llm):
        """Well-grounded answers should score ≥ 0.6 on faithfulness."""
        metric = FaithfulnessMetric(threshold=0.6, model=eval_llm, async_mode=False)
        failures = []
        for qa in SAMPLE_QA[:2]:
            case = _make_test_case(qa)
            metric.measure(case)
            if metric.score < 0.6:
                failures.append(f"{qa['question'][:50]}: {metric.score:.2f}")
        assert not failures, f"Low faithfulness scores: {failures}"

    def test_hallucinated_answer_detected(self, eval_llm):
        """Claims about MySQL/encryption are not in the pgvector context."""
        case = _make_test_case(BAD_ANSWER_CASE)
        metric = FaithfulnessMetric(threshold=0.5, model=eval_llm, async_mode=False)
        metric.measure(case)
        assert metric.score < 0.5, (
            f"Expected low faithfulness for hallucinated answer, got {metric.score:.2f}"
        )


# ===========================================================================
# 3. Contextual Precision
#    Checks that the most relevant chunks are ranked highest.
# ===========================================================================

class TestContextualPrecision:
    def test_pgvector_context_precision(self, eval_llm):
        case = _make_test_case(SAMPLE_QA[0])
        metric = ContextualPrecisionMetric(threshold=0.6, model=eval_llm, async_mode=False)
        assert_test(case, [metric])

    def test_retrieval_context_precision(self, eval_llm):
        case = _make_test_case(SAMPLE_QA[1])
        metric = ContextualPrecisionMetric(threshold=0.6, model=eval_llm, async_mode=False)
        assert_test(case, [metric])


# ===========================================================================
# 4. Contextual Recall
#    Checks that the retrieved context covers all key points in the ground truth.
# ===========================================================================

class TestContextualRecall:
    def test_pgvector_context_recall(self, eval_llm):
        case = _make_test_case(SAMPLE_QA[0])
        metric = ContextualRecallMetric(threshold=0.6, model=eval_llm, async_mode=False)
        assert_test(case, [metric])

    def test_retrieval_context_recall(self, eval_llm):
        case = _make_test_case(SAMPLE_QA[1])
        metric = ContextualRecallMetric(threshold=0.6, model=eval_llm, async_mode=False)
        assert_test(case, [metric])


# ===========================================================================
# 5. Contextual Relevancy
#    Checks that each retrieved chunk is relevant to the question.
# ===========================================================================

class TestContextualRelevancy:
    def test_pgvector_context_relevancy(self, eval_llm):
        case = _make_test_case(SAMPLE_QA[0])
        metric = ContextualRelevancyMetric(threshold=0.6, model=eval_llm, async_mode=False)
        assert_test(case, [metric])


# ===========================================================================
# 6. Multi-metric combined test
# ===========================================================================

class TestCombinedMetrics:
    @pytest.mark.flaky(reruns=1)
    def test_full_rag_quality(self, eval_llm):
        """Best-case sample scored across all five RAG quality dimensions."""
        case = _make_test_case(SAMPLE_QA[0])
        metrics = [
            AnswerRelevancyMetric(threshold=0.5, model=eval_llm, async_mode=False),
            FaithfulnessMetric(threshold=0.6, model=eval_llm, async_mode=False),
            ContextualPrecisionMetric(threshold=0.6, model=eval_llm, async_mode=False),
            ContextualRecallMetric(threshold=0.6, model=eval_llm, async_mode=False),
            ContextualRelevancyMetric(threshold=0.6, model=eval_llm, async_mode=False),
        ]
        assert_test(case, metrics)


# ===========================================================================
# 7. Integration tests (require live DB + API)
# ===========================================================================

@pytest.mark.integration
class TestIntegration:
    """
    Integration tests that call the real RAG pipeline against the live DB.

    The knowledge base contains Wikivoyage travel content (Haikou and other
    Chinese cities), so questions are chosen from that domain.

    Run with: pytest tests/test_deepeval_rag.py -m integration
    Requires: OPENROUTER_API_KEY and PGVECTOR_DB_URL in .env (both are set).
    """

    def _rag_answer(self, question: str):
        """Retrieve pgvector context and answer via LlamaIndex LLM — no LangChain."""
        from scraper.raq_query import retrieve_top3, format_docs
        from llama_index.llms.openai import OpenAI as LlamaOpenAI
        from llama_index.llms.openai.utils import ALL_AVAILABLE_MODELS
        from llama_index.core.llms import ChatMessage, MessageRole

        model = os.getenv("OPENAI_FREE_MODEL", "openai/gpt-oss-20b:free")
        if model not in ALL_AVAILABLE_MODELS:
            ALL_AVAILABLE_MODELS[model] = 128000

        docs = retrieve_top3(question)
        context = format_docs(docs)
        contexts = [d.page_content for d in docs]

        llm = LlamaOpenAI(
            model=model,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            api_base="https://openrouter.ai/api/v1",
            temperature=0.1,
            max_tokens=4096,
        )
        response = llm.chat([ChatMessage(
            role=MessageRole.USER,
            content=(
                "You are a helpful assistant. Use the following context to answer the question.\n\n"
                f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            ),
        )])
        return response.message.content or "", contexts

    @pytest.mark.flaky(reruns=2)
    def test_rag_chain_faithfulness(self, eval_llm):
        """RAG chain answer about Haikou should be grounded in retrieved context."""
        question = "How can I get to Haikou by plane?"
        answer, contexts = self._rag_answer(question)
        assert contexts, "Knowledge base returned no documents — check PGVECTOR_DB_URL"
        assert answer, "LLM returned an empty answer"

        case = LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=contexts,
        )
        metric = FaithfulnessMetric(threshold=0.5, model=eval_llm, async_mode=False)
        assert_test(case, [metric])

    @pytest.mark.flaky(reruns=2)
    def test_rag_chain_answer_relevancy(self, eval_llm):
        """RAG chain answer should directly address a question about Haikou."""
        question = "What is Haikou known for?"
        answer, contexts = self._rag_answer(question)
        assert contexts, "Knowledge base returned no documents — check PGVECTOR_DB_URL"
        assert answer, "LLM returned an empty answer"

        case = LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=contexts,
        )
        metric = AnswerRelevancyMetric(threshold=0.5, model=eval_llm, async_mode=False)
        assert_test(case, [metric])

    @pytest.mark.flaky(reruns=2)
    def test_rag_chain_contextual_recall(self, eval_llm):
        """Retrieved context should cover key facts about Haikou as a destination."""
        question = "What is Haikou and where is it located?"
        ground_truth = "Haikou is the capital of Hainan province in China."
        answer, contexts = self._rag_answer(question)
        assert contexts, "Knowledge base returned no documents — check PGVECTOR_DB_URL"
        assert answer, "LLM returned an empty answer"

        case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=ground_truth,
            retrieval_context=contexts,
        )
        metrics = [
            ContextualRecallMetric(threshold=0.5, model=eval_llm, async_mode=False),
            ContextualRelevancyMetric(threshold=0.5, model=eval_llm, async_mode=False),
        ]
        assert_test(case, metrics)

    @pytest.mark.flaky(reruns=2)
    def test_llamaindex_agent_local_knowledge(self, eval_llm):
        """LlamaIndex agent should use search_local_knowledge to answer a travel question."""
        import asyncio
        from scraper.llamaindex_agent import aquery_agent

        question = "What should I know about visiting Haikou as a tourist?"
        result = asyncio.run(aquery_agent(question))
        answer = result.get("answer", "")
        assert answer, "Agent returned an empty answer"

        case = LLMTestCase(input=question, actual_output=answer)
        metric = AnswerRelevancyMetric(threshold=0.5, model=eval_llm, async_mode=False)
        assert_test(case, [metric])
