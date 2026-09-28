import csv
import json
import os
import random
import re
from pathlib import Path


def read_topics(path):
    topics = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            topics[parts[0]] = parts[1]
    return topics


def read_qrels(path):
    qrels = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _, docid, rel = parts[:4]
            qrels.setdefault(qid, []).append((docid, int(rel)))
    return qrels


def format_doc_text(doc):
    title = doc.get("title", "") or ""
    text = doc.get("text", "") or ""
    title = title.strip()
    text = text.strip()
    if title and text:
        return title + "\n" + text
    return title or text


def save_jsonl(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stream_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_corpus_docs(corpus_dir, needed_docids, max_docs=None):
    needed_docids = set(needed_docids)
    docs = {}
    loaded = 0
    files = sorted(Path(corpus_dir).glob("*.jsonl"))
    for file_path in files:
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                docid = obj.get("docid") or obj.get("id")
                if docid in needed_docids:
                    docs[docid] = {
                        "docid": docid,
                        "title": obj.get("title", "") or "",
                        "text": obj.get("text", "") or "",
                    }
                    loaded += 1
                    if max_docs and loaded >= max_docs:
                        return docs
                    if loaded >= len(needed_docids):
                        return docs
    return docs


def get_all_qrels_docids(*qrels_maps):
    docids = set()
    for qrels in qrels_maps:
        for items in qrels.values():
            for docid, _ in items:
                docids.add(docid)
    return docids


def build_pairs(qrels, topics, docs, negative_ratio=1, include_all_negatives=False, random_seed=42):
    random.seed(random_seed)
    rows = []
    for qid, items in qrels.items():
        query = topics.get(qid)
        if query is None:
            continue
        positives = [docid for docid, rel in items if rel >= 1]
        negatives = [docid for docid, rel in items if rel == 0]
        if not positives:
            continue
        for docid in positives:
            if docid not in docs:
                continue
            rows.append({"query_id": qid, "query": query, "docid": docid, "doc": format_doc_text(docs[docid]), "label": 1})
        if not negatives:
            continue
        if include_all_negatives:
            sampled_negatives = negatives
        else:
            negative_count = len(positives) * negative_ratio
            sampled_negatives = random.sample(negatives, min(len(negatives), negative_count))
        for docid in sampled_negatives:
            if docid not in docs:
                continue
            rows.append({"query_id": qid, "query": query, "docid": docid, "doc": format_doc_text(docs[docid]), "label": 0})
    return rows


def tokenize_text(text):
    text = text.lower()
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    return tokens


def load_corpus_jsonl(path):
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            docs.append(json.loads(line))
    return docs
