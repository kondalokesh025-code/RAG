import math
import statistics
import time

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from tokenization import bm25_preprocess


def evaluate_retrieval(chunked_documents, vector_store, rerank):
    K = 5

    # Test queries and their expected source files
        
    queries = [
        {
            "input": "Where is the main Signer class implemented?",
            "expected_sources": ["src/itsdangerous/signer.py"]
        },
        {
            "input": "How does ItsDangerous generate HMAC signatures?",
            "expected_sources": ["src/itsdangerous/signer.py"]
        },
        {
            "input": "How are signing keys derived from secret keys and salts?",
            "expected_sources": ["src/itsdangerous/signer.py"]
        },
        {
            "input": "How does Signer create and verify signatures?",
            "expected_sources": ["src/itsdangerous/signer.py"]
        },
        {
            "input": "How does ItsDangerous support secret key rotation?",
            "expected_sources": ["src/itsdangerous/signer.py", "src/itsdangerous/serializer.py"]
        },
        {
            "input": "How does Serializer serialize and sign arbitrary Python objects?",
            "expected_sources": ["src/itsdangerous/serializer.py"]
        },
        {
            "input": "How does Serializer verify signatures and deserialize payloads?",
            "expected_sources": ["src/itsdangerous/serializer.py", "src/itsdangerous/signer.py"]
        },
        {
            "input": "How are serialized payloads converted into bytes?",
            "expected_sources": ["src/itsdangerous/serializer.py", "src/itsdangerous/encoding.py"]
        },
        {
            "input": "How does ItsDangerous handle invalid payloads and signatures?",
            "expected_sources": ["src/itsdangerous/exc.py", "src/itsdangerous/serializer.py"]
        },
        {
            "input": "How does TimestampSigner attach timestamps to signatures?",
            "expected_sources": ["src/itsdangerous/timed.py"]
        },
        {
            "input": "How does ItsDangerous validate signature expiration using max_age?",
            "expected_sources": ["src/itsdangerous/timed.py", "src/itsdangerous/exc.py"]
        },
        {
            "input": "How does TimedSerializer use TimestampSigner?",
            "expected_sources": ["src/itsdangerous/timed.py", "src/itsdangerous/serializer.py"]
        },
        {
            "input": "How does URLSafeSerializer encode and decode serialized data?",
            "expected_sources": ["src/itsdangerous/url_safe.py", "src/itsdangerous/encoding.py"]
        },
        {
            "input": "How does ItsDangerous compress data before URL-safe encoding?",
            "expected_sources": ["src/itsdangerous/url_safe.py"]
        },
        {
            "input": "How does URLSafeTimedSerializer combine URL-safe serialization with expiration?",
            "expected_sources": ["src/itsdangerous/url_safe.py", "src/itsdangerous/timed.py"]
        },
        {
            "input": "Signer",
            "expected_sources": ["src/itsdangerous/signer.py"]
        },
        {
            "input": "Serializer",
            "expected_sources": ["src/itsdangerous/serializer.py"]
        },
        {
            "input": "TimestampSigner",
            "expected_sources": ["src/itsdangerous/timed.py"]
        },
        {
            "input": "URLSafeSerializer",
            "expected_sources": ["src/itsdangerous/url_safe.py"]
        },
        {
            "input": "SignatureExpired",
            "expected_sources": ["src/itsdangerous/exc.py"]
        },
        {
            "input": "HMACAlgorithm",
            "expected_sources": ["src/itsdangerous/signer.py"]
        },
        {
            "input": "base64_decode",
            "expected_sources": ["src/itsdangerous/encoding.py"]
        },
        {
            "input": "TimedSerializer",
            "expected_sources": ["src/itsdangerous/timed.py"]
        }
    ]


    def normalize_source(source):
        return str(source).replace("\\\\", "/")

    def matches_source(source, expected):
        return source == expected or source.endswith("/" + expected)

    # Build standalone and hybrid retrievers for comparison.
    chroma = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": K}
    )

    bm25 = BM25Retriever.from_documents(
        chunked_documents,
        preprocess_func=bm25_preprocess,
        k=K
    )

    hybrid = EnsembleRetriever(
        retrievers=[chroma, bm25],
        weights=[0.5, 0.5],
        c=60
    )

    methods = [
        ("Chroma", chroma.invoke),
        ("BM25", bm25.invoke),
        ("Hybrid", hybrid.invoke),
        ("Hybrid + Reranker", rerank)
    ]

    summaries = {}
    reranker_details = []

    for name, retrieve in methods:
        hits = 0
        recall_sum = 0.0
        rr_sum = 0.0
        latencies = []

        failed_queries = 0
        empty_queries = 0
        total_documents = 0
        documents_without_source = 0
        duplicate_source_documents = 0
        documents_with_source = 0
        total_unique_sources = 0
        too_many_results = 0

        for item in queries:
            query = item["input"]
            expected = [
                normalize_source(source)
                for source in item["expected_sources"]
            ]

            start_time = time.perf_counter()

            try:
                results = retrieve(query)
                results = list(results) if results is not None else []

                if len(results) > K:
                    too_many_results += 1

                results = results[:K]

                if any(not hasattr(doc, "metadata") for doc in results):
                    raise TypeError("Retriever returned an invalid document.")

            except Exception as error:
                failed_queries += 1
                results = []
                print(
                    f"\nERROR [{name}] query: {query}\n"
                    f"{type(error).__name__}: {error}"
                )

            latency = time.perf_counter() - start_time
            latencies.append(latency)

            if not results:
                empty_queries += 1

            sources = []

            for doc in results:
                source = normalize_source(doc.metadata.get("source", ""))

                if source:
                    sources.append(source)
                else:
                    documents_without_source += 1

            total_documents += len(results)
            documents_with_source += len(sources)

            unique_sources = set(sources)
            total_unique_sources += len(unique_sources)
            duplicate_source_documents += len(sources) - len(unique_sources)

            ranks = [
                rank
                for rank, source in enumerate(sources, start=1)
                if any(matches_source(source, path) for path in expected)
            ]

            matched_sources = {
                path
                for path in expected
                if any(matches_source(source, path) for source in sources)
            }

            hit = int(bool(ranks))
            recall = len(matched_sources) / len(expected) if expected else 0
            reciprocal_rank = 1 / ranks[0] if ranks else 0

            hits += hit
            recall_sum += recall
            rr_sum += reciprocal_rank

            if name == "Hybrid + Reranker":
                reranker_details.append({
                    "query": query,
                    "hit": hit,
                    "recall": recall,
                    "mrr": reciprocal_rank,
                    "rank": ranks[0] if ranks else None,
                    "latency_ms": latency * 1000,
                    "sources": sources
                })

        count = len(queries)
        p95_index = max(0, math.ceil(0.95 * len(latencies)) - 1)
        sorted_latencies = sorted(latencies)

        duplicate_rate = (
            duplicate_source_documents / documents_with_source
            if documents_with_source else 0
        )

        summaries[name] = {
            "hit_rate": hits / count,
            "source_recall": recall_sum / count,
            "mrr": rr_sum / count,
            "avg_latency_ms": statistics.mean(latencies) * 1000,
            "median_latency_ms": statistics.median(latencies) * 1000,
            "p95_latency_ms": sorted_latencies[p95_index] * 1000,
            "max_latency_ms": max(latencies) * 1000,
            "failed_queries": failed_queries,
            "empty_queries": empty_queries,
            "too_many_results": too_many_results,
            "duplicate_source_rate": duplicate_rate,
            "missing_source_documents": documents_without_source,
            "average_unique_sources": total_unique_sources / count
        }

    print(f"\n{'Method':<22}{'Hit@5':>10}{'SourceRecall@5':>20}{'MRR@5':>10}{'Latency ms':>13}")
    print("-" * 75)

    for name, result in summaries.items():
        print(
            f"{name:<22}"
            f"{result['hit_rate']:>10.4f}"
            f"{result['source_recall']:>20.4f}"
            f"{result['mrr']:>10.4f}"
            f"{result['avg_latency_ms']:>13.2f}"
        )

    rerank_summary = summaries["Hybrid + Reranker"]

    print("\n========== RERANKER DIAGNOSTICS ==========")
    print(f"Queries tested: {len(queries)}")
    print(f"Failed queries: {rerank_summary['failed_queries']}")
    print(f"Empty results: {rerank_summary['empty_queries']}")
    print(f"Queries returning more than {K} documents: {rerank_summary['too_many_results']}")
    print(f"Median query latency: {rerank_summary['median_latency_ms']:.2f} ms")
    print(f"95th-percentile query latency: {rerank_summary['p95_latency_ms']:.2f} ms")
    print(f"Maximum query latency: {rerank_summary['max_latency_ms']:.2f} ms")
    print(f"Average unique source files per query: {rerank_summary['average_unique_sources']:.2f}")
    print(f"Duplicate-source document rate: {rerank_summary['duplicate_source_rate']:.2%}")
    print(f"Documents missing source metadata: {rerank_summary['missing_source_documents']}")

    print("\n========== RERANKER QUERY DETAILS ==========")
    for detail in reranker_details:
        rank = detail["rank"] if detail["rank"] is not None else "Not found"
        print(f"\nQuery: {detail['query']}")
        print(f"Hit@5: {detail['hit']}")
        print(f"SourceRecall@5: {detail['recall']:.4f}")
        print(f"MRR contribution: {detail['mrr']:.4f}")
        print(f"First relevant rank: {rank}")
        print(f"Latency: {detail['latency_ms']:.2f} ms")
        for source in detail["sources"]:
            print(f"  Source: {source}")

        print("\n========== RETRIEVAL EVALUATION ==========")
    print(f"Queries evaluated: {len(queries)}")
    print(f"Results cutoff: {K}")

    headers = (
        f"{'Method':<22}"
        f"{'Hit@5':>9}"
        f"{'Recall@5':>11}"
        f"{'MRR@5':>9}"
        f"{'Avg ms':>10}"
        f"{'P95 ms':>10}"
        f"{'Dup %':>9}"
    )

    print("\n" + headers)
    print("-" * len(headers))

    for name, result in summaries.items():
        print(
            f"{name:<22}"
            f"{result['hit_rate']:>9.4f}"
            f"{result['source_recall']:>11.4f}"
            f"{result['mrr']:>9.4f}"
            f"{result['avg_latency_ms']:>10.2f}"
            f"{result['p95_latency_ms']:>10.2f}"
            f"{result['duplicate_source_rate']:>8.2%}"
        )

    print("\n========== RERANKER DIAGNOSTICS ==========")

    result = summaries["Hybrid + Reranker"]

    print(f"Failed queries: {result['failed_queries']}")
    print(f"Empty results: {result['empty_queries']}")
    print(f"Queries returning more than {K} documents: {result['too_many_results']}")
    print(f"Median latency: {result['median_latency_ms']:.2f} ms")
    print(f"Maximum latency: {result['max_latency_ms']:.2f} ms")
    print(f"Average unique source files per query: {result['average_unique_sources']:.2f}")
    print(f"Duplicate-source document rate: {result['duplicate_source_rate']:.2%}")
    print(f"Documents missing source metadata: {result['missing_source_documents']}")

    print("\n========== RERANKER QUERY DETAILS ==========")

    for detail in reranker_details:
        rank = detail["rank"] if detail["rank"] is not None else "Not found"

        print(f"\nQuery: {detail['query']}")
        print(f"Hit@5: {detail['hit']}")
        print(f"SourceRecall@5: {detail['recall']:.4f}")
        print(f"MRR contribution: {detail['mrr']:.4f}")
        print(f"First relevant rank: {rank}")
        print(f"Latency: {detail['latency_ms']:.2f} ms")

        for source in detail["sources"]:
            print(f"  Source: {source}")

    return summaries