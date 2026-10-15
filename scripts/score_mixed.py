#!/usr/bin/env python3
"""Score all mixed previews and retain clearly labelled illustrative examples."""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from t2m.encoder import encode_text, make_encoder


def words(value):
    return re.findall(r"[a-z0-9]+", value.lower())


def token_f1(prediction, reference):
    from collections import Counter
    a, b = Counter(words(prediction)), Counter(words(reference))
    overlap = sum((a & b).values())
    if not a or not b or not overlap:
        return 0.0
    precision, recall = overlap / sum(a.values()), overlap / sum(b.values())
    return 2 * precision * recall / (precision + recall)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", required=True)
    parser.add_argument("--eval-cache", default="data/cache/eval_mixed")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    preview = Path(args.preview)
    rows = [json.loads(line) for line in preview.open(encoding="utf8")]
    if len(rows) != 80:
        raise ValueError(f"Expected all 80 fixed evaluation items, found {len(rows)}")
    output = Path(args.out) if args.out else preview.parent.parent / "analysis" / preview.stem
    output.mkdir(parents=True, exist_ok=True)

    encoder = make_encoder(only_text=True)
    generated = [r["generated_text"].strip() or " " for r in rows]
    generated_emb = encode_text(encoder, generated, batch_size=16)
    all_refs = []
    refs_per_row = []
    for row in rows:
        refs = row.get("references") or [row.get("reference_text", "")]
        if not refs or not any(refs):
            raise ValueError(f"Missing evaluation reference for {row['path']}")
        refs_per_row.append(list(range(len(all_refs), len(all_refs) + len(refs))))
        all_refs.extend(refs)
    reference_emb = encode_text(encoder, all_refs, batch_size=16)
    by_modality = defaultdict(list)
    cache = Path(args.eval_cache)
    source_vectors = {}
    for kind in ("image", "audio", "video", "text"):
        selected = [i for i, row in enumerate(rows) if row["modality"] == kind]
        vectors = np.load(cache / f"{kind}_embeddings.npy").astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(1e-12)
        if len(selected) != len(vectors):
            raise ValueError(f"{kind}: preview/cache count mismatch")
        source_vectors[kind] = vectors
        sims = generated_emb[selected] @ vectors.T
        for j, i in enumerate(selected):
            row = rows[i]
            refs = row.get("references") or [row["reference_text"]]
            values = reference_emb[refs_per_row[i]] @ generated_emb[i]
            item = {**row,
                    "source_cosine": round(float(sims[j, j]), 4),
                    "source_retrieval_rank": int(1 + (sims[j] > sims[j, j]).sum()),
                    "reference_cosine": round(float(values.max()), 4),
                    "reference_token_f1": round(max(token_f1(row["generated_text"], ref)
                                                    for ref in refs), 4),
                    "generated_tokens_approx": len(words(row["generated_text"]))}
            by_modality[kind].append(item)

    all_items = [item for kind in ("image", "audio", "video", "text") for item in by_modality[kind]]
    with (output / "per_item.jsonl").open("w", encoding="utf8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = {"preview": str(preview), "count": len(rows), "max_new_tokens": rows[0]["max_new_tokens"],
               "metric_note": "Embedding cosine and within-modality retrieval use the same frozen EmbeddingGemma 2 encoder; image references are five human captions, video references are coarse action labels."}
    for kind, items in by_modality.items():
        summary[kind] = {"count": len(items),
                         "mean_reference_cosine": round(float(np.mean([x["reference_cosine"] for x in items])), 4),
                         "mean_source_cosine": round(float(np.mean([x["source_cosine"] for x in items])), 4),
                         "source_retrieval_at_1": round(float(np.mean([x["source_retrieval_rank"] == 1 for x in items])), 4),
                         "mean_reference_token_f1": round(float(np.mean([x["reference_token_f1"] for x in items])), 4),
                         "mean_generated_words": round(float(np.mean([x["generated_tokens_approx"] for x in items])), 1)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")

    selected = []
    for kind, items in by_modality.items():
        good = sorted(items, key=lambda x: (x["reference_cosine"], x["reference_token_f1"]), reverse=True)[:4]
        failure = min(items, key=lambda x: x["reference_cosine"])
        selected.extend([{**item, "selection_reason": "top_reference_similarity_within_modality"}
                         for item in good])
        selected.append({**failure, "selection_reason": "illustrative_failure"})
    with (output / "selected_examples.jsonl").open("w", encoding="utf8") as f:
        for item in selected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps({"summary": str(output / "summary.json"),
                      "selected": str(output / "selected_examples.jsonl"),
                      "modalities": {k: summary[k] for k in by_modality}}, indent=2))


if __name__ == "__main__":
    main()
