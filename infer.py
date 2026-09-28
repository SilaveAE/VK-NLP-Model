import argparse
import json
import logging
import math
import re
from pathlib import Path

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from utils import load_corpus_jsonl, tokenize_text


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def build_index(corpus_jsonl):
    docs = []
    collection = []
    with open(corpus_jsonl, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            docs.append(item)
            collection.append(tokenize_text(item["text"]))
    bm25 = BM25Okapi(collection)
    return docs, bm25


def rerank_query(model, query, candidates, top_k, batch_size=32):
    if not candidates:
        return []
    pair_texts = [[query, c["text"]] for c in candidates]
    scores = []
    for i in range(0, len(pair_texts), batch_size):
        scores.extend(model.predict(pair_texts[i : i + batch_size]))
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Answer a query with BM25 retrieval and reranking.")
    parser.add_argument("--model-path", required=True, help="Path to trained reranker model")
    parser.add_argument("--corpus-jsonl", required=True, help="Path to prepared corpus JSONL")
    parser.add_argument("--query", required=True, help="User query text")
    parser.add_argument("--top-k", type=int, default=5, help="Number of answers to return")
    parser.add_argument("--bm25-k", type=int, default=50, help="Number of BM25 candidates to retrieve")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for reranking")
    args = parser.parse_args()

    logging.info("Loading BM25 corpus from %s", args.corpus_jsonl)
    docs, bm25 = build_index(args.corpus_jsonl)

    query_tokens = tokenize_text(args.query)
    scores = bm25.get_scores(query_tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: args.bm25_k]
    candidates = [docs[idx] for idx in top_indices]

    model = CrossEncoder(args.model_path)
    logging.info("Reranking %d BM25 candidates with model %s", len(candidates), args.model_path)
    ranked = rerank_query(model, args.query, candidates, args.top_k, batch_size=args.batch_size)

    for rank, (doc, score) in enumerate(ranked, start=1):
        print(f"=== Rank {rank} | score={score:.4f} | docid={doc['docid']} ===")
        print(doc["text"])
        print()


if __name__ == "__main__":
    main()
