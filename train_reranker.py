import argparse
import json
import logging
import math
import os
import random

from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_examples(path, max_samples=None):
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            examples.append(InputExample(texts=[item["query"], item["doc"]], label=float(item["label"])))
            if max_samples and len(examples) >= max_samples:
                break
    random.shuffle(examples)
    return examples


def main():
    parser = argparse.ArgumentParser(description="Train a cross-encoder reranker on MIRACL data.")
    parser.add_argument("--train-pairs", required=True, help="Path to train_pairs.jsonl")
    parser.add_argument("--dev-pairs", required=True, help="Path to dev_pairs.jsonl")
    parser.add_argument("--output-dir", default="models/reranker", help="Output directory for trained model")
    parser.add_argument("--model-name", default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="Hugging Face model name for CrossEncoder")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Maximum number of samples to load from train pairs")
    parser.add_argument("--max-dev-samples", type=int, default=2000, help="Maximum number of samples to load from dev pairs")
    parser.add_argument("--evaluation-steps", type=int, default=1000, help="Evaluation frequency in training steps")
    parser.add_argument("--device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu", help="Compute device to train on")
    args = parser.parse_args()

    logging.info("Loading train examples from %s", args.train_pairs)
    train_examples = load_examples(args.train_pairs, max_samples=args.max_train_samples)
    logging.info("Loaded %d train examples", len(train_examples))

    logging.info("Loading dev examples from %s", args.dev_pairs)
    dev_examples = load_examples(args.dev_pairs, max_samples=args.max_dev_samples)
    logging.info("Loaded %d dev examples", len(dev_examples))

    model = CrossEncoder(args.model_name, num_labels=1, max_length=256, device=args.device)

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    warmup_steps = math.ceil(len(train_dataloader) * args.epochs * 0.1)
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info("Training with %d warmup steps", warmup_steps)
    logging.info("Skipping in-training evaluator for CrossEncoder. Use evaluate.py after training.")

    model.fit(
        train_dataloader=train_dataloader,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        output_path=args.output_dir,
        use_amp=False,
    )

    logging.info("Ensuring model is saved to %s", args.output_dir)
    model.save(args.output_dir)

    saved_files = []
    for root, dirs, files in os.walk(args.output_dir):
        for f in files:
            saved_files.append(os.path.relpath(os.path.join(root, f), args.output_dir))

    logging.info("Training finished. Saved %d file(s) under %s", len(saved_files), args.output_dir)
    for saved_file in saved_files:
        logging.info(" - %s", saved_file)


if __name__ == "__main__":
    main()
