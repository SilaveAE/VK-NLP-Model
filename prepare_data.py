import argparse
import json
import logging
import os
from pathlib import Path

from utils import (build_pairs, format_doc_text, get_all_qrels_docids,
                   load_corpus_docs, read_qrels, read_topics, save_jsonl)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MIRACL train/dev data and corpus for reranker training."
    )
    parser.add_argument("--corpus-dir", required=True, help="Path to corpus JSONL files")
    parser.add_argument("--topics-train", required=True, help="Path to train topics TSV")
    parser.add_argument("--qrels-train", required=True, help="Path to train qrels TSV")
    parser.add_argument("--topics-dev", required=True, help="Path to dev topics TSV")
    parser.add_argument("--qrels-dev", required=True, help="Path to dev qrels TSV")
    parser.add_argument("--output-dir", default="data", help="Output directory for prepared files")
    parser.add_argument("--negative-ratio", type=int, default=1, help="Negative examples per positive example")
    parser.add_argument("--max-docs", type=int, default=None, help="Maximum number of corpus documents to load")
    parser.add_argument("--include-all-negatives", action="store_true", help="Include all negative qrels examples")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading topics and qrels")
    topics_train = read_topics(args.topics_train)
    qrels_train = read_qrels(args.qrels_train)
    topics_dev = read_topics(args.topics_dev)
    qrels_dev = read_qrels(args.qrels_dev)

    logging.info("Collecting relevant document IDs from train and dev qrels")
    all_docids = get_all_qrels_docids(qrels_train, qrels_dev)
    logging.info("Total unique qrels document IDs: %d", len(all_docids))

    logging.info("Loading matching documents from corpus")
    docs = load_corpus_docs(args.corpus_dir, all_docids, max_docs=args.max_docs)
    logging.info("Loaded documents: %d", len(docs))

    corpus_rows = []
    for docid, doc in docs.items():
        corpus_rows.append({"docid": docid, "text": format_doc_text(doc)})
    save_jsonl(corpus_rows, output_dir / "corpus.jsonl")
    logging.info("Saved corpus JSONL to %s", output_dir / "corpus.jsonl")

    logging.info("Building train pairs")
    train_pairs = build_pairs(
        qrels_train,
        topics_train,
        docs,
        negative_ratio=args.negative_ratio,
        include_all_negatives=args.include_all_negatives,
    )
    save_jsonl(train_pairs, output_dir / "train_pairs.jsonl")
    logging.info("Saved train pairs to %s", output_dir / "train_pairs.jsonl")

    logging.info("Building dev pairs")
    dev_pairs = build_pairs(
        qrels_dev,
        topics_dev,
        docs,
        negative_ratio=args.negative_ratio,
        include_all_negatives=args.include_all_negatives,
    )
    save_jsonl(dev_pairs, output_dir / "dev_pairs.jsonl")
    logging.info("Saved dev pairs to %s", output_dir / "dev_pairs.jsonl")

    logging.info("Prepared all files in %s", output_dir)


if __name__ == "__main__":
    main()
