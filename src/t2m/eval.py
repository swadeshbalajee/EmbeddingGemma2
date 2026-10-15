"""Deterministic validation and human-inspectable media JSONL previews."""
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from .data import collate, different_indices


def _mean_metrics(scores):
    keys = scores[0].keys()
    return {k: float(sum(s[k] for s in scores)/len(scores)) for k in keys}


@torch.inference_mode()
def evaluate_text(model, dataset, bos_id, pad_id, device, batch_size=16, n_batches=8, seed=1234):
    model.eval()
    rng = np.random.default_rng(seed)
    limit = min(len(dataset), batch_size*n_batches)
    if limit < 2:
        return {}
    ix = rng.choice(len(dataset), size=limit, replace=False)
    loader = DataLoader(Subset(dataset, ix.tolist()), batch_size=batch_size, shuffle=False,
                        collate_fn=lambda items: collate(items,bos_id,pad_id),drop_last=True)
    result = []
    for batch in loader:
        batch = {k:v.to(device, non_blocking=True) for k,v in batch.items()}
        z, x, y, mask = (batch[k] for k in ("embeddings","input_ids","labels","mask"))
        # Deranged negatives are different records; fixed RNG makes it reproducible.
        negative = torch.arange(len(z), device=device).roll(1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type=="cuda" and torch.cuda.is_bf16_supported()):
            pos, pos_tok, counts = model.token_logprobs(z, x,y,mask)
            shuffled, _, _ = model.token_logprobs(z[negative], x,y,mask)
            zero, _, _ = model.token_logprobs(torch.zeros_like(z), x,y,mask)
        # In-batch nearest-neighbor embeddings provide harder validation negatives.
        sims = F.normalize(z.float(), dim=-1) @ F.normalize(z.float(), dim=-1).T
        sims.fill_diagonal_(-float("inf"))
        hard_idx = sims.argmax(dim=-1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type=="cuda" and torch.cuda.is_bf16_supported()):
            hard, _, _ = model.token_logprobs(z[hard_idx], x,y,mask)
        delta = pos-shuffled
        v = {
            "ce":float(-pos.mean()), "ppl":float(math.exp(min(20.,float(-pos.mean())))),
            "delta_cond":float(delta.mean()), "rank_acc":float((delta>0).float().mean()),
            "delta_zero":float((pos-zero).mean()),
            "hard_negative_delta":float((pos-hard).mean()),
            "hard_negative_rank_acc":float((pos>hard).float().mean()),
            "positive_logprob":float(pos.mean()),
            "shuffled_logprob":float(shuffled.mean()),
            "zero_logprob":float(zero.mean()),
        }
        # Correct - shuffled conditional NLL difference on early and late positions.
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type=="cuda" and torch.cuda.is_bf16_supported()):
            _, neg_tok, _ = model.token_logprobs(z[negative], x,y,mask)
        early_mask = y[:,:8].ne(-100)
        late_mask = y[:,8:].ne(-100)
        v["early_delta_cond"] = float(((neg_tok[:,:8]-pos_tok[:,:8])*early_mask).sum()/early_mask.sum().clamp(min=1))
        v["late_delta_cond"] = float(((neg_tok[:,8:]-pos_tok[:,8:])*late_mask).sum()/late_mask.sum().clamp(min=1))
        result.append(v)
    model.train()
    return _mean_metrics(result) if result else {}


@torch.inference_mode()
def evaluate_media(model, tokenizer, eval_cache, output_dir, step, variant,
                   device, limit=32, max_new_tokens=96, repetition_penalty=1.08):
    """Read precomputed modality embeddings and generate JSONL with img/path -> text."""
    root = Path(eval_cache)
    dest = Path(output_dir) / "previews"
    dest.mkdir(parents=True,exist_ok=True)
    records = []
    bos = tokenizer.bos_token_id
    if bos is None:
        bos = tokenizer.eos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    if bos is None or eos is None:
        raise ValueError("Decoder tokenizer requires BOS or EOS")
    for kind in ("image", "audio", "video", "text"):
        emb_file = root / f"{kind}_embeddings.npy"
        manifest_file = root / f"{kind}_manifest.jsonl"
        if not (emb_file.exists() and manifest_file.exists()):
            continue
        embeddings = np.load(emb_file, mmap_mode="r")
        with manifest_file.open(encoding="utf8") as f:
            manifest = [json.loads(s) for s in f]
        if len(manifest) != len(embeddings):
            raise ValueError(f"{kind} eval cache mismatch")
        for j in range(min(len(embeddings), limit)):
            z = torch.tensor(np.array(embeddings[j],dtype=np.float32),device=device).unsqueeze(0)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type=="cuda" and torch.cuda.is_bf16_supported()):
                output_ids = model.generate_from_embedding(z, bos, eos, pad,
                              max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty)
            ids = output_ids[0].tolist()
            ended_with_eos = eos in ids
            generated_token_count = ids.index(eos) + 1 if ended_with_eos else len(ids)
            generated = tokenizer.decode(ids, skip_special_tokens=True).strip()
            record = {"step":step, "variant":variant, "modality":kind,
                      "path":manifest[j]["path"], "image":manifest[j]["path"] if kind=="image" else None,
                      "generated_text":generated,
                      "generated_token_count":generated_token_count,
                      "ended_with_eos":ended_with_eos,
                      "max_new_tokens":max_new_tokens, "decode":"greedy"}
            for key in ("reference_text", "references", "reference_type", "action_label",
                        "source_split", "dataset", "source"):
                if manifest[j].get(key) is not None:
                    record[key] = manifest[j][key]
            records.append(record)
    if not records:
        raise FileNotFoundError(f"No eval media caches under {root}. Run 'python -m t2m.cache_media ...' first.")
    output_file = dest / f"step_{step:07d}.jsonl"
    with output_file.open("w", encoding="utf8") as f:
        for record in records:
            f.write(json.dumps(record,ensure_ascii=False)+"\n")
    model.train()
    return str(output_file), len(records)
