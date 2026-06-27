"""
Ragas test suite for the Agentic-RAG pipeline.

Covers:
  - faithfulness          – answers grounded in retrieved context?
  - answer_relevancy      – answers address the question?
  - context_precision     – relevant chunks ranked highest?
  - context_recall        – retrieved context covers expected answer?

Test modes
----------
Static (no real DB needed):
  Uses pre-built QA pairs from conftest.SAMPLE_QA.
  Requires OPENROUTER_API_KEY for the LLM judge.

Integration (marked @pytest.mark.integration):
  Calls the live retriever and rag_chain.
  Requires a running PostgreSQL/pgvector DB AND OPENROUTER_API_KEY.
  Run with:  pytest tests/test_ragas_rag.py -m integration
"""

import os
import warnings
import pytest

warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv

load_dotenv()

from tests.conftest import SAMPLE_QA, BAD_ANSWER_CASE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ragas_llm():
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set – skipping Ragas tests")

    model = os.getenv("OPENAI_FREE_MODEL", "openai/gpt-oss-20b:free")
    lc_llm = ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0,
    )
    return LangchainLLMWrapper(lc_llm)


@pytest.fixture(scope="session")
def ragas_embeddings():
    """Local sentence-transformers embeddings — avoids async connection-pool issues."""
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_huggingface import HuggingFaceEmbeddings

    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return LangchainEmbeddingsWrapper(emb)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample(qa: dict):
    from ragas.dataset_schema import SingleTurnSample
    return SingleTurnSample(
        user_input=qa["question"],
        response=qa["answer"],
        retrieved_contexts=qa["contexts"],
        reference=qa["ground_truth"],
    )


def _dataset(samples):
    from ragas.dataset_schema import EvaluationDataset
    return EvaluationDataset(samples=samples)


def _evaluate(dataset, metrics, run_config=None) -> dict:
    """Run ragas evaluate and return a dict of {metric_name: avg_score}."""
    import math
    from ragas import evaluate
    from ragas.run_config import RunConfig

    if run_config is None:
        run_config = RunConfig(timeout=60, max_retries=3, max_wait=30)
    raw = evaluate(dataset=dataset, metrics=metrics, run_config=run_config)
    scores_list = raw.scores
    if not scores_list:
        return {}

    keys = scores_list[0].keys()
    result = {}
    for k in keys:
        vals = [
            s[k] for s in scores_list
            if k in s and s[k] is not None and not math.isnan(s[k])
        ]
        result[k] = sum(vals) / len(vals) if vals else float("nan")
    return result


def _get_metrics(ragas_llm, ragas_embeddings=None, which=("faith",)):
    """Return fresh, configured metric objects for each call."""
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    import copy

    result = {}
    if "faith" in which:
        m = copy.deepcopy(faithfulness)
        m.llm = ragas_llm
        result["faithfulness"] = m
    if "relevancy" in which:
        m = copy.deepcopy(answer_relevancy)
        m.llm = ragas_llm
        if ragas_embeddings:
            m.embeddings = ragas_embeddings
        result["answer_relevancy"] = m
    if "precision" in which:
        m = copy.deepcopy(context_precision)
        m.llm = ragas_llm
        result["context_precision"] = m
    if "recall" in which:
        m = copy.deepcopy(context_recall)
        m.llm = ragas_llm
        result["context_recall"] = m
    return result


# ===========================================================================
# 1. Faithfulness
# ===========================================================================

class TestFaithfulness:
    def test_faithful_answers(self, ragas_llm):
        """First two samples should score ≥ 0.6 on faithfulness."""
        metrics = _get_metrics(ragas_llm, which=("faith",))
        result = _evaluate(
            _dataset([_make_sample(s) for s in SAMPLE_QA[:2]]),
            list(metrics.values()),
        )
        assert result["faithfulness"] >= 0.6, f"faithfulness={result['faithfulness']:.2f}"

    def test_hallucinated_answer_low_score(self, ragas_llm):
        """MySQL-encryption answer to a pgvector question should score low."""
        metrics = _get_metrics(ragas_llm, which=("faith",))
        result = _evaluate(_dataset([_make_sample(BAD_ANSWER_CASE)]), list(metrics.values()))
        assert result["faithfulness"] < 0.5, (
            f"Expected low faithfulness for hallucinated answer, got {result['faithfulness']:.2f}"
        )


# ===========================================================================
# 2. Answer Relevancy
# ===========================================================================

class TestAnswerRelevancy:
    @pytest.mark.flaky(reruns=1)
    def test_relevant_answers(self, ragas_llm, ragas_embeddings):
        """First two samples should score ≥ 0.6 on answer relevancy."""
        metrics = _get_metrics(ragas_llm, ragas_embeddings, which=("relevancy",))
        result = _evaluate(
            _dataset([_make_sample(s) for s in SAMPLE_QA[:2]]),
            list(metrics.values()),
        )
        assert result["answer_relevancy"] >= 0.6, f"answer_relevancy={result['answer_relevancy']:.2f}"


# ===========================================================================
# 3. Context Precision
# ===========================================================================

class TestContextPrecision:
    @pytest.mark.flaky(reruns=1)
    def test_pgvector_context_precision(self, ragas_llm):
        metrics = _get_metrics(ragas_llm, which=("precision",))
        result = _evaluate(_dataset([_make_sample(SAMPLE_QA[0])]), list(metrics.values()))
        assert result["context_precision"] >= 0.5, f"context_precision={result['context_precision']:.2f}"

    @pytest.mark.flaky(reruns=1)
    def test_retrieval_context_precision(self, ragas_llm):
        metrics = _get_metrics(ragas_llm, which=("precision",))
        result = _evaluate(_dataset([_make_sample(SAMPLE_QA[1])]), list(metrics.values()))
        assert result["context_precision"] >= 0.5, f"context_precision={result['context_precision']:.2f}"


# ===========================================================================
# 4. Context Recall
# ===========================================================================

class TestContextRecall:
    def test_pgvector_context_recall(self, ragas_llm):
        metrics = _get_metrics(ragas_llm, which=("recall",))
        result = _evaluate(_dataset([_make_sample(SAMPLE_QA[0])]), list(metrics.values()))
        assert result["context_recall"] >= 0.6, f"context_recall={result['context_recall']:.2f}"

    def test_retrieval_context_recall(self, ragas_llm):
        metrics = _get_metrics(ragas_llm, which=("recall",))
        result = _evaluate(_dataset([_make_sample(SAMPLE_QA[1])]), list(metrics.values()))
        assert result["context_recall"] >= 0.6, f"context_recall={result['context_recall']:.2f}"


# ===========================================================================
# 5. Multi-metric combined evaluation
# ===========================================================================

class TestCombinedMetrics:
    @pytest.mark.flaky(reruns=1)
    def test_all_metrics_on_best_case(self, ragas_llm, ragas_embeddings):
        """All four Ragas metrics should pass on the pgvector QA pair."""
        metrics = _get_metrics(
            ragas_llm, ragas_embeddings,
            which=("faith", "relevancy", "precision", "recall"),
        )
        result = _evaluate(_dataset([_make_sample(SAMPLE_QA[0])]), list(metrics.values()))

        threshold = 0.6
        failures = [f"{k}={v:.2f}" for k, v in result.items() if v < threshold]
        assert not failures, f"Metrics below {threshold}: {failures}"


# ===========================================================================
# 6. Integration tests (require live DB + API)
# ===========================================================================

@pytest.mark.integration
class TestIntegration:
    """
    End-to-end Ragas evaluation using the live RAG pipeline.

    Run with:  pytest tests/test_ragas_rag.py -m integration
    Requires: OPENROUTER_API_KEY and PGVECTOR_DB_URL in .env (both are set).
    """

    def _build_test_chain(self):
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough
        from scraper.raq_query import retrieve_top3, format_docs

        llm = ChatOpenAI(
            model=os.getenv("OPENAI_FREE_MODEL", "openai/gpt-oss-20b:free"),
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.1,
        )
        prompt = ChatPromptTemplate.from_template(
            "You are a helpful assistant. Use the following context to answer the question.\n\n"
            "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        )
        return (
            {"context": lambda x: format_docs(retrieve_top3(x)), "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )

    def _live_run_config(self):
        from ragas.run_config import RunConfig
        return RunConfig(timeout=150, max_retries=5, max_wait=60)

    @pytest.mark.flaky(reruns=2)
    def test_rag_chain_faithfulness(self, ragas_llm):
        from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
        from scraper.raq_query import retrieve_top3

        question = "How can I get to Haikou by plane?"
        docs = retrieve_top3(question)
        assert docs, "Knowledge base returned no documents — check PGVECTOR_DB_URL"
        contexts = [d.page_content for d in docs]
        chain = self._build_test_chain()
        answer = chain.invoke(question)

        sample = SingleTurnSample(
            user_input=question, response=answer, retrieved_contexts=contexts,
            reference="Haikou has an international airport that serves domestic and international flights.",
        )
        metrics = _get_metrics(ragas_llm, which=("faith",))
        result = _evaluate(EvaluationDataset(samples=[sample]), list(metrics.values()), self._live_run_config())
        assert result["faithfulness"] >= 0.5, f"faithfulness={result['faithfulness']:.2f}"

    @pytest.mark.flaky(reruns=2)
    def test_rag_chain_answer_relevancy(self, ragas_llm, ragas_embeddings):
        """Live RAG answer should be relevant to the question asked."""
        from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
        from scraper.raq_query import retrieve_top3

        question = "What is Haikou known for as a tourist destination?"
        docs = retrieve_top3(question)
        assert docs, "Knowledge base returned no documents — check PGVECTOR_DB_URL"
        contexts = [d.page_content for d in docs]
        chain = self._build_test_chain()
        answer = chain.invoke(question)

        sample = SingleTurnSample(
            user_input=question, response=answer, retrieved_contexts=contexts,
        )
        metrics = _get_metrics(ragas_llm, ragas_embeddings, which=("relevancy",))
        result = _evaluate(EvaluationDataset(samples=[sample]), list(metrics.values()), self._live_run_config())
        assert result["answer_relevancy"] >= 0.5, f"answer_relevancy={result['answer_relevancy']:.2f}"

    @pytest.mark.flaky(reruns=2)
    def test_rag_chain_full_pipeline(self, ragas_llm, ragas_embeddings):
        from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
        from scraper.raq_query import retrieve_top3

        question = "What is there to do in Haikou?"
        ground_truth = "Haikou is a city in Hainan, China with various attractions and activities."
        docs = retrieve_top3(question)
        assert docs, "Knowledge base returned no documents — check PGVECTOR_DB_URL"
        contexts = [d.page_content for d in docs]
        chain = self._build_test_chain()
        answer = chain.invoke(question)

        sample = SingleTurnSample(
            user_input=question, response=answer, retrieved_contexts=contexts, reference=ground_truth,
        )
        faith_metrics = _get_metrics(ragas_llm, which=("faith",))
        faith_result = _evaluate(EvaluationDataset(samples=[sample]), list(faith_metrics.values()), self._live_run_config())
        rel_metrics = _get_metrics(ragas_llm, ragas_embeddings, which=("relevancy",))
        rel_result = _evaluate(EvaluationDataset(samples=[sample]), list(rel_metrics.values()), self._live_run_config())

        result = {**faith_result, **rel_result}
        print("\n=== Live pipeline Ragas scores ===")
        for k, v in result.items():
            print(f"  {k}: {v:.3f}")

        assert result["faithfulness"] >= 0.5, f"faithfulness={result['faithfulness']:.2f}"
        assert result["answer_relevancy"] >= 0.5, f"answer_relevancy={result['answer_relevancy']:.2f}"

    @pytest.mark.flaky(reruns=2)
    def test_langgraph_agent_local_knowledge(self, ragas_llm, ragas_embeddings):
        from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
        from scraper.langgraph_agent import query_custom_agent

        question = "What should I know about visiting Haikou as a tourist?"
        agent_result = query_custom_agent(question)
        answer = agent_result.get("answer", "")
        assert answer, "Agent returned an empty answer"

        sample = SingleTurnSample(user_input=question, response=answer, retrieved_contexts=[answer])
        metrics = _get_metrics(ragas_llm, ragas_embeddings, which=("relevancy",))
        result = _evaluate(EvaluationDataset(samples=[sample]), list(metrics.values()), self._live_run_config())
        assert result["answer_relevancy"] >= 0.5, f"answer_relevancy={result['answer_relevancy']:.2f}"
