from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader


def recall_at_k(sim_matrix: np.ndarray, k: int):
    if k <= 0:
        raise ValueError("k must be positive")
    if sim_matrix.ndim != 2:
        raise ValueError("sim_matrix must be 2-D")
    k = min(k, sim_matrix.shape[1])
    top_k = np.argpartition(sim_matrix, -k, axis=1)[:, -k:]
    hits = (top_k == np.arange(sim_matrix.shape[0])[:, None]).any(axis=1)
    return hits.mean()


def evaluate_model(model, lang_texts, candidates):
    metrics: Dict[str, Dict[str, float]] = {}
    cand_emb = model.encode(
        candidates,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    for lang, texts in lang_texts.items():
        qembed = model.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        sim = qembed @ cand_emb.T
        metrics[lang] = {
            "R@1": recall_at_k(sim, 1),
            "R@5": recall_at_k(sim, 5),
            "R@10": recall_at_k(sim, 10),
        }
    return metrics


def pretty_print(tag, scores, langs):
    print(tag)
    for lang in langs:
        lang_scores = scores[lang]
        summary = ", ".join(f"{metric}: {value:.3f}" for metric, value in lang_scores.items())
        print(f"  {lang}-en | {summary}")


def main():
    data_path = Path("sample-6lang.jsonl")
    langs = ["ar", "de", "el", "fr", "zh"]
    eval_docs = 1000
    train_batch_size = 32
    fine_tune_epochs = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    articles: List[Dict[str, str]] = []
    entries: Dict[str, Dict[str, str]] = defaultdict(dict)
    with data_path.open(mode="r", encoding="utf-8") as source:
        for line in source:
            rec = json.loads(line)
            articles.append(rec)
            entries[rec["id"]][rec["lang"]] = rec["text"]

    query_articles = [rec["text"] for rec in articles if rec["lang"] == "zh"]
    result_articles = [rec["text"] for rec in articles if rec["lang"] == "en"]
    print(f"Loaded {len(articles)} rows, {len(result_articles)} English docs, {len(query_articles)} zh docs.")

    needed_langs = set(langs + ["en"])
    ids_with_all_langs = [doc_id for doc_id in sorted(entries) if needed_langs.issubset(entries[doc_id])]
    if len(ids_with_all_langs) < eval_docs + 1:
        raise ValueError("not enough multilingual articles for the requested split")

    eval_ids = ids_with_all_langs[:eval_docs]
    train_ids = ids_with_all_langs[eval_docs:]

    lang_texts = {lang: [] for lang in langs}
    candidates: List[str] = []
    remaining = set(langs)
    for doc_id in eval_ids:
        record = entries[doc_id]
        candidates.append(record["en"])
        for lang in langs:
            bucket = lang_texts[lang]
            if len(bucket) < eval_docs:
                bucket.append(record[lang])
                if len(bucket) == eval_docs:
                    remaining.discard(lang)
        if not remaining:
            break

    training_examples: List[InputExample] = []
    for doc_id in train_ids:
        record = entries[doc_id]
        english = record.get("en")
        if not english:
            continue
        for lang in langs:
            other = record.get(lang)
            if other:
                training_examples.append(InputExample(texts=[english, other]))

    print(
        f"{len(candidates)} eval english docs; {len(training_examples)} "
        "parallel sentence pairs reserved for fine-tuning."
    )
    print(f"Training device: {device}")

    labse = SentenceTransformer("sentence-transformers/LaBSE", device=device)

    # quick-and-dirty sanity check copied from notebook explorations
    qembed = labse.encode(query_articles[:200])
    rembed = labse.encode(result_articles[:500])
    sim = qembed @ rembed.T
    argmax = np.argmax(sim, axis=1)
    print(f"First sanity argmax slice: {argmax[:5]}")

    baseline_scores = evaluate_model(labse, lang_texts, candidates)
    pretty_print("Baseline LaBSE", baseline_scores, langs)

    train_loader = DataLoader(training_examples, batch_size=train_batch_size, shuffle=True)
    loss = losses.MultipleNegativesRankingLoss(labse)
    warmup_steps = max(1, int(len(train_loader) * fine_tune_epochs * 0.1))
    labse.fit(
        train_objectives=[(train_loader, loss)],
        epochs=fine_tune_epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
    )

    tuned_scores = evaluate_model(labse, lang_texts, candidates)
    pretty_print("Fine-tuned LaBSE", tuned_scores, langs)


if __name__ == "__main__":
    main()
