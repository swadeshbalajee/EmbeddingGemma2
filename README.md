# text2modal-v1 — frozen multimodal encoder → text-only prefix-conditioned generator

**Experiment:** Can a *single* decoder trained **only on text embeddings** describe a previously unseen image/audio/video after replacing the input embedding with another modality's pooled embedding?

**V1-B (default):** Single 768→576 continuous prefix + SmolLM2-135M + positive CE and correct-vs-shuffled *conditional ranking loss*.

**V1-C:** Identical architecture and losses, but independently add small Gaussian noise to both positive and shuffled embeddings, followed by L2 normalization.

The encoder (`google/embeddinggemma-2`) stays **frozen**, including mean pooling and its final 768d projection. Its **text model is ~270M parameters; full multimodal model is ~740M**. The cached 768d vectors are produced with no task prefix. The decoder is `HuggingFaceTB/SmolLM2-135M` (base checkpoint), not an instruction-tuned model.

> **Current study:** The original V1-B run and the 200K three-variant pilot have completed on two RTX 3090s. The 1M comparison is in progress. See [docs/progress.md](docs/progress.md) for live results, run status, literature, and limitations; [docs/comparison_200k.md](docs/comparison_200k.md) for the completed pilot comparison.

## Current research comparison

The three text-only decoder training variants are one-token prefix with ranking loss (V1-B), one-token prefix with ranking and Gaussian embedding noise, and eight-token prefix with ranking and the same noise. The encoder is frozen in every case. The 1M training mixture has 300K longer ShareGPT4V image descriptions and 150K short Conceptual Captions, plus audio, video, and prose text. The fixed mixed evaluation contains 20 real examples each of image, audio, video, and text. Full final previews use greedy decoding with a 128-token limit and live under `runs/<variant>/previews/`; scoring and selected good and failed examples live under `runs/<variant>/analysis/`.

Run logs with tqdm are in `logs/`. The stage journal is `logs/research_pipeline.jsonl`; watcher JSONLs under `logs/` record steps, GPU use, and free disk. See `configs/exp_*` for the exact 200K and 1M settings.

## Installation: Ubuntu 24.04, 2×RTX 3090, CUDA

```bash
conda create -n text2modal python=3.11 -y
conda activate text2modal
# Match Torch with the CUDA drivers on your workstation. Example CUDA 12.8 wheel:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
python -m pytest -q
```

`environment.yml` is also supplied; from this directory, `conda env create -f environment.yml && conda activate text2modal && pip install -e .` is an alternative. The pinned requirement ranges allow Torch's CUDA build already selected above to remain installed. `torch.compile` is optional because cold compilation may be slower than eager execution for short smoke runs. Native PyTorch SDPA selects FlashAttention kernels where supported, without a mandatory `flash-attn` source build.

You may need `huggingface-cli login` (or `hf auth login`) and acceptance of any Hugging Face model terms before downloading weights. Disk space must cover decoder/model downloads, text caches, training checkpoints, and the dataset files.

## Quick start (first experiment)

```bash
# 1. Build 200K mixed examples + held-out set; download TEXT only.
python -m t2m.mix --out data/mixed --train-size 200000 --val-size 2000

# 2. Embed with frozen EmbeddingGemma 2, cache as numpy memory-mapped arrays.
python -m t2m.cache_text --input data/mixed/train.jsonl --out data/cache/train --max-seq-len 256 --batch-size 32
python -m t2m.cache_text --input data/mixed/val.jsonl   --out data/cache/val   --max-seq-len 256 --batch-size 32

# 3. Demo images are only for software-path verification, NOT real caption quality.
python scripts/prepare_demo_images.py
python -m t2m.cache_media --kind image --input-dir data/eval_images --out data/cache/eval_media

# 4. Two 3090s: 20-step training/evaluation check.
bash scripts/benchmark.sh

# 5. Full V1-B run (default) after checking smoke output.
bash scripts/train_v1b.sh

# 6. Full V1-C run, using the SAME cached data and mixed evaluation set.
bash scripts/train_v1c.sh
```

Training and benchmark console output, including tqdm progress bars, is appended to `logs/benchmark.log`, `logs/v1b.log`, and `logs/v1c.log`. Structured metrics remain in each run's `metrics.jsonl` file.

The checked-in evaluation preparation script downloads a fixed set of 20 Flickr30k images, 20 MInDS-14 English audio clips, and 20 UCF101 videos from Hugging Face. It also selects 20 FineWeb-Edu passages from the held-out prose validation data. It writes raw examples and provenance to `data/eval_mixed/manifest.jsonl`, then the encoder cache to `data/cache/eval_mixed`:

```bash
python scripts/prepare_eval_mixed.py
python -m t2m.cache_mixed_eval --manifest data/eval_mixed/manifest.jsonl --out data/cache/eval_mixed
```

`configs/v1b.yaml` points at this cache; V1-C inherits the same path. Each preview therefore contains up to 20 examples from each of the four modalities. Generated rows keep the existing fields and include source/reference text when available. To evaluate a saved checkpoint without starting another training run:

```bash
python scripts/eval_checkpoint.py --config configs/v1b.yaml --checkpoint runs/v1b/checkpoints/final.pt
```

This writes a step-numbered JSONL file under the run's `previews/` directory. Audio/video encoding needs the media dependencies required by Transformers (in the current environment: `librosa` and `av`).

Audio/video (eval only):

```bash
python -m t2m.cache_media --kind audio --input-dir data/my_audio --out data/cache/eval_media
python -m t2m.cache_media --kind video --input-dir data/my_videos --out data/cache/eval_media
```

The frozen full multimodal encoder is loaded only when caching evaluation media. Encoding video may need the media dependencies expected by Transformers (e.g. `av`, `decord` or ffmpeg), depending on backend. Training does not require these modalities.

## Mixture: text data only

| Source | Portion | Notes |
|:--|--:|:--|
| Conceptual Captions 3M | 35% | Image alt-text captions; images never fetched |
| WavCaps | 25% | Audio event descriptions from JSON metadata; no sound fetched |
| WebVid-10M | 20% | Video action description text; videos never fetched |
| FineWeb-Edu | 20% | Coherent prose paragraphs (rather than random word windows) |

Text is cleaned and deduplicated globally by normalized string, with deterministic hash-based held-out sampling. If a source is missing, quota is not silently filled by another source: the builder stops unless `--allow-shortfall` is explicitly provided. WavCaps is effectively academic/non-commercial; WebVid's hosting has AGPL conditions; check original content and license restrictions before redistribution or non-research use. For controlled or custom sources, pass `--local-image path.jsonl --local-audio path.jsonl --local-video path.jsonl --local-prose path.jsonl`. Each JSONL record has a `text` or `caption` field. These are text-only records, not multimodal training pairs.

For larger experiments:

```bash
python -m t2m.mix --out data/mixed_1m --train-size 1000000 --val-size 10000
python -m t2m.cache_text --input data/mixed_1m/train.jsonl --out data/cache/train_1m
python -m t2m.cache_text --input data/mixed_1m/val.jsonl --out data/cache/val_1m
# Edit train_cache and val_cache in config to the new directories before training.
```

**10M:** the 25% WavCaps quota cannot be filled uniquely from its ~403K total captions. Add more audio-description sources or choose new mixture quotas. Never silently repeat texts. Frozen-encoder embedding preprocessing can be slower than the first decoder training run and is excluded from training-time estimates.

## Exact conditioning and loss

Let `z = normalize(EmbeddingGemma2(text))` be one `[768]` vector and `p = PrefixMLP(z)` one `[576]` continuous prefix. Feed the pretrained causal decoder the embeddings `[p, BOS, y1, ..., y_(T-1)]`. Its logits **at BOS** predict `y1`, not BOS; the prefix has no language-model label. Length budget 256 includes 1 prefix + 1 BOS + up to 254 target tokens (including EOS). Dynamic padding ensures short captions do not do unnecessary attention work.

With `s(y,z) = (1/T) sum_t log P(y_t|y_<t, p(z))`, choose an incorrect in-batch embedding `z-` while keeping teacher-forced target text **identical**:

```
CE   = mean_i( -s(y_i,z_i) )
Rank = mean_i softplus( margin + s(y_i,z_i^-) - s(y_i,z_i) )
Loss = CE + lambda * Rank
```

Defaults: `margin=0.2`, `lambda=0.2`. Negatives are random non-identity cyclic permutations within the local GPU microbatch. The text dataset is deduplicated; semantically identical paraphrases are **still possible**, so a future hard-negative stage should filter semantic equivalences. Both positive and negative forwards have gradients. While the LM is frozen in adapter warmup, gradients still propagate **through** its operations to the prefix adapter.

**V1-C example config:** both embeddings receive independent `N(0,0.005^2)` per-component perturbations and are renormalized during training. The current controlled noise variants use `0.012` as recorded in `configs/exp_*`; validation and generation turn noise off. Noise magnitude is a tunable hypothesis, not an established ideal.

## Optimizations

- Torchrun + NCCL **DDP** (2 GPUs, not `DataParallel`), gradient accumulation, distributed sampler.
- **BF16 activations** via autocast (RTX 3090), model parameters float32; fused AdamW optimizer; TF32 matmuls.
- PyTorch **SDPA**, selects flash attention backends where supported; prefix generation uses `past_key_values` (KV caching).
- Encoder is never loaded during decoder training, only during offline embedding cache creation.
- Training caches in `numpy` memory-mapped arrays, pin_memory DataLoader, dynamic padding.
- `torch_compile: true` and `gradient_checkpointing: true` optional YAML toggles; compile first-call overhead and kernel availability depend on environment. Avoid gradient checkpointing unless required to fit memory.
- Adapter-only warmup (default 500 steps), then joint decoder fine-tuning. Decoder lr 2e-5, adapter lr 2e-4, cosine scheduler, warmup, clipping.

**Caveats:** Both variants run a positive *and negative* model forward for each microbatch, increasing compute. Peak memory/throughput varies greatly with actual target length. Use `configs/smoke.yaml` to measure actual throughput instead of relying on speculative hours.

## Intermediate evaluations and artifacts

`runs/v1b/metrics.jsonl`: training CE/ranking/total, condition delta, rank and margin accuracy, gradient norm, throughput, GPU memory, and validation perplexity/deltas from correct, shuffled and **zeroed** embeddings. `val/delta_cond = mean(s(y,z_correct)-s(y,z_shuffled))` on held-out text. A second in-batch nearest-neighbor (cosine) negative provides `val/hard_negative_delta` and `val/hard_negative_rank_acc`. Also tracks early/late token conditional advantage and `delta_zero`.

`runs/v1b/previews/step_0000250.jsonl`: generated text for each evaluation input. Example schema:

```json
{"step":250,"variant":"v1b","modality":"image","image":"/path/to/photo.png","path":"/path/to/photo.png","generated_text":"A red car parked near a tree.","max_new_tokens":96,"decode":"greedy"}
```

The historical V1-B preview limit was 96 new tokens. Its final preview and all final study comparisons were rerun at 128. Intermediate schedules depend on the config; `max_seq_len=256` is a maximum *training input* length, not a requirement to produce 256 output tokens. You can change `max_new_tokens` independently.

A model could distinguish correct-vs-shuffled **teacher-forced** embeddings and nevertheless generate hallucinated or generic captions. Always manually inspect full generations on held-out images, and compare baseline V1-B with V1-C. No evaluator in this code claims image captions are objectively correct.

## Resuming

Checkpoints include model, optimizer, scheduler, step, epoch, and batch cursor:

```bash
# YAML only: set resume: runs/v1b/checkpoints/step_0001000.pt
bash scripts/train_v1b.sh
```

The train sampler is deterministically reshuffled by epoch. Resuming at an optimizer-step boundary is supported; external dataset caches must not change. The `adapter_warmup_steps` boundary is omitted as a periodic checkpoint if it exactly coincides with the warmup-to-joint optimizer reinitialization. RNG state is not bitwise synchronized across DDP ranks on restore, so resumed training is reproducible approximately, not bitwise.

## Limits / known risks

- **Scientific**: the pretrained multimodal encoder already learned image/audio/video representations. The claim is **text-only decoder training**, never "no multimodal pretraining."
- The pooled embedding is semantic, not invertible; perfect verbatim text reconstruction and fine-grained captioning are not expected.
- The conditional ranking objective can still exploit teacher forcing and in-distribution artifacts; it does not force zero-shot cross-modal success.
- `torch.compile`, FlashAttention selection, HF model APIs, and real end-to-end performance must be verified on your two 3090 GPUs. Local tests cover tensor alignment and logic but not external downloaded model contracts.
- WavCaps and WebVid dataset downloads can be slow; use the `--local-*` inputs if necessary.

## References / source models

- Encoder: https://huggingface.co/google/embeddinggemma-2
- Decoder: https://huggingface.co/HuggingFaceTB/SmolLM2-135M
- Dataset: https://huggingface.co/datasets/google-research-datasets/conceptual_captions
- Dataset: https://huggingface.co/datasets/cvssp/WavCaps
- Dataset: https://huggingface.co/datasets/TempoFunk/webvid-10M
- Dataset: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
