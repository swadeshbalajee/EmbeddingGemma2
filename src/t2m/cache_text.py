import argparse
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from .encoder import make_encoder, encode_text


def cache(input_file, output, encoder_id, decoder_id, max_seq_len=256, batch_size=32):
    path = Path(input_file)
    if not path.is_file():
        raise FileNotFoundError(path)
    n = sum(1 for _ in path.open(encoding="utf8"))
    if n == 0:
        raise ValueError("Empty dataset")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "metadata.json"
    if meta_path.exists():
        raise FileExistsError(f"Cache {out} exists; remove it to rebuild")
    tokenizer = AutoTokenizer.from_pretrained(decoder_id)
    if tokenizer.eos_token_id is None:
        raise ValueError("Decoder tokenizer must have EOS")
    encoder = make_encoder(encoder_id, only_text=True)
    emb = np.lib.format.open_memmap(out / "embeddings.npy", mode="w+", dtype=np.float16, shape=(n,768))
    tok = np.lib.format.open_memmap(out / "tokens.npy", mode="w+", dtype=np.int32, shape=(n,max_seq_len-2))
    lens = np.lib.format.open_memmap(out / "lengths.npy", mode="w+", dtype=np.uint16, shape=(n,))

    def store(batch, offset):
        texts = [v["text"] for v in batch]
        vectors = encode_text(encoder, texts, batch_size=batch_size)
        tokenized = tokenizer(texts, add_special_tokens=False, truncation=True, max_length=max_seq_len-3)["input_ids"]
        for j, ids in enumerate(tokenized):
            ids = ids[:max_seq_len-3] + [tokenizer.eos_token_id]
            # total positions: 1 prefix + 1 BOS + n target <= max_seq_len.
            i = offset+j
            emb[i] = vectors[j].astype(np.float16)
            tok[i, :] = -1
            tok[i, :len(ids)] = np.array(ids, dtype=np.int32)
            lens[i] = len(ids)

    buf = []
    seen = 0
    with path.open(encoding="utf8") as f:
        for line in f:
            row = json.loads(line)
            buf.append(row)
            if len(buf) >= batch_size:
                store(buf, seen)
                seen += len(buf)
                buf = []
                if seen % (batch_size*50) == 0:
                    print(f"Cached {seen}/{n}", flush=True)
    if buf:
        store(buf, seen)
        seen += len(buf)
    assert seen == n
    emb.flush(); tok.flush(); lens.flush()
    meta = {"count": n, "max_seq_len": max_seq_len, "encoder_id": encoder_id,
            "decoder_id": decoder_id, "embedding_dim":768, "pooling":"official normalized mean",
            "text_prompt":"", "input_file": str(path)}
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Created {out} for {n} examples")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--encoder-id", default="google/embeddinggemma-2")
    p.add_argument("--decoder-id", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    a = p.parse_args()
    cache(a.input,a.out,a.encoder_id,a.decoder_id,a.max_seq_len,a.batch_size)

if __name__ == "__main__":
    main()
