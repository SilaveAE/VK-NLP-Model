import argparse
import json
import logging
import statistics
from pathlib import Path

from sentence_transformers import CrossEncoder
from utils import read_qrels, read_topics, load_corpus_jsonl


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_corpus_map(corpus_jsonl):
    docs = {}
    with open(corpus_jsonl, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            docs[item["docid"]]=item["text"]
    return docs


def chunked(iterable, chunk_size):
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i:i+chunk_size]


def main():
    parser = argparse.ArgumentParser(description="Evaluate reranker on MIRACL dev qrels.")
    parser.add_argument("--model-path", required=True, help="Path to trained reranker model")
    parser.add_argument("--qrels-dev", required=True, help="Path to dev qrels TSV")
    parser.add_argument("--topics-dev", required=True, help="Path to dev topics TSV")
    parser.add_argument("--corpus-jsonl", required=True, help="Path to prepared corpus JSONL")
    parser.add_argument("--top-k", type=int, default=10, help="Cutoff for top-k metrics")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for scoring")
    args = parser.parse_args()

    logging.info("Loading corpus from %s", args.corpus_jsonl)
    docs = load_corpus_map(args.corpus_jsonl)
    logging.info("Loaded %d documents", len(docs))

    logging.info("Loading dev topics and qrels")
    topics = read_topics(args.topics_dev)
    qrels = read_qrels(args.qrels_dev)

    model = CrossEncoder(args.model_path)
    logging.info("Loaded model from %s", args.model_path)

    mrrs = []
    p1 = []
    topk_hits = []
    skipped = 0

    for qid, items in qrels.items():
        query = topics.get(qid)
        if query is None:
            continue
        candidates = []
        relevant_docids = set()
        for docid, rel in items:
            if docid not in docs:
                continue
            candidates.append((docid, docs[docid], rel))
            if rel >= 1:
                relevant_docids.add(docid)
        if not candidates or not relevant_docids:
            skipped += 1
            continue

        pair_texts = [[query, doc_text] for _, doc_text, _ in candidates]
        scores = []
        for batch in chunked(pair_texts, args.batch_size):
            scores.extend(model.predict(batch))

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        rank = 1
        found_rank = None
        for (docid, _, rel), _ in ranked:
            if rel >= 1:
                found_rank = rank
                break
            rank += 1
        if found_rank is None:
            mrrs.append(0.0)
            p1.append(0.0)
            topk_hits.append(0.0)
            continue

        mrrs.append(1.0 / found_rank)
        p1.append(1.0 if found_rank == 1 else 0.0)
        topk_hits.append(1.0 if found_rank <= args.top_k else 0.0)

    total = len(qrels) - skipped
    logging.info("Evaluated %d queries, skipped %d queries with missing data", total, skipped)
    logging.info("MRR@%d = %.4f", args.top_k, statistics.mean(mrrs) if mrrs else 0.0)
    logging.info("P@1 = %.4f", statistics.mean(p1) if p1 else 0.0)
    logging.info("Top-%d Hit = %.4f", args.top_k, statistics.mean(topk_hits) if topk_hits else 0.0)


if __name__ == "__main__":
    main()
