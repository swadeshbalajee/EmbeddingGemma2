# Poster notes: text-only decoding of frozen multimodal embeddings

Status: working draft; the three final 1M runs will determine the headline comparison. See `docs/progress.md` for the execution record.

The first two 1M runs are complete. One-prefix noise improves the image correct-minus-zero control from +0.625 to +1.274 nats/token, while their combined 60-media-item reference-cosine difference is only +0.0016 with a paired 95% interval crossing zero. The eight-prefix 1M run is training, so this is an interim observation.

## Research question

Can a small causal decoder trained **only on text embeddings** describe images, audio, and video when supplied the frozen EmbeddingGemma 2 pooled embedding? This studies transfer through an existing multimodal representation; the encoder itself had multimodal pretraining.

## Three practical variants

| Variant | Conditioning | Training change |
|---|---|---|
| One prefix | Project 768-d pooled vector to one 576-d prefix token | CE plus correct-versus-shuffled ranking |
| One prefix + noise | Same one-token prefix | Gaussian embedding noise, σ = 0.012, before normalization |
| Eight prefixes + noise | Project pooled vector to eight 576-d prefix tokens | Same noise and losses |

All three use the same frozen encoder, SmolLM2-135M base decoder, text-only source types, rank objective, and fixed 80-item mixed evaluation. Eight prefix tokens change positions and attention competition as well as learned capacity, so the comparison tests the full prefix design, not capacity alone.

The adapter is a 768→1,024→(576 × K) MLP with GELU and one learned output scale. It has 1,377,857 parameters for K=1 and 5,510,657 for K=8; both variants fine-tune the same decoder after a 500-step adapter-only warmup. This parameter difference is part of the practical design comparison.

## Evaluation panel

The fixed set contains 20 real examples per modality: Flickr30k images, MInDS-14 spoken audio, UCF101 video clips, and FineWeb-Edu text. Use the 128-token greedy outputs in each final `previews/step_*.jsonl`. Score generated text against five human captions per image, transcripts for audio, coarse action labels for video, and original passages for text. Report the encoder-based reference cosine as a *semantic diagnostic* and token F1 as a complementary lexical check; inspect examples and failures. The video labels are not detailed captions.

Teacher-forced correct, shuffled, and zero embedding likelihoods measure conditioning dependence. A controlled 1,024-item text validation set measures text reconstruction. All full outputs, per-item scores, and explicitly selected examples are retained under each run's `analysis/` folder.

## Initial 200K pilot result

| Variant | Image cosine | Audio cosine | Video cosine | Held-out text CE |
|---|---:|---:|---:|---:|
| One prefix | 0.705 | 0.698 | 0.709 | 2.182 |
| One prefix + noise | 0.740 | 0.710 | 0.722 | 2.252 |
| Eight prefixes + noise | **0.753** | **0.765** | 0.720 | **1.916** |

The eight-prefix pilot beats the one-prefix baseline by +0.0423 reference cosine on the 60 media items; a paired 10,000-resample bootstrap stratified by modality gives [+0.0250, +0.0600]. Its video-only interval crosses zero. These small fixed-set exploratory intervals and same-encoder scoring do not establish broad caption quality. Full details: `docs/comparison_200k.md` and `runs/analysis/paired_prefix8_200k_vs_v1b.json`.

Pilot figure: `docs/figures/mixed_200k.png` (source: `scripts/plot_mixed_comparison.py`). The final 1M version will use all three completed 1M runs.

## Unusual behavior worth showing

1. **Media-specific early stopping.** The original V1-B emits EOS for all 60 media items after mean 6.3 image, 10.3 audio, and 6.9 video tokens, despite a 128-token allowance. Text inputs average 75.2 generated tokens. The 1M baseline still emits EOS on all media items, even with 300K longer image descriptions. Figure: `runs/v1b/analysis/step_0003124/output_lengths.png` and the corresponding final 1M figures. This reveals a transfer problem hidden by low text CE.
2. **Image zero-input anomaly.** For V1-B, the correct image embedding beats the shuffled image embedding for 17/20 images (mean +1.662 nats/token), yet zero embedding beats correct on all 20 (correct-minus-zero −1.598 nats/token). This is a calibration diagnostic, not a causal explanation: the zero vector lies outside the encoder's normalized output support, and the caption source differs from training. The 1M baseline improves correct-minus-zero to +0.625. Figure: `runs/v1b/analysis/step_0003124/conditioning_deltas.png`.

## Qualitative panel candidates

Curated cross-run cases, including successes and errors, are in `runs/analysis/case_studies_200k_vs_1m.jsonl`. Use the actual image/audio/video file and human reference beside the model output. State clearly that the selected cases illustrate behavior and are not representative accuracy. Update the panel with the final 1M outputs before presenting.

## Constraints and claims

- The pooled 768-d embedding is compressed and may lack fine-grained detail. The decoder cannot reconstruct information absent from it.
- Image captions, audio transcripts, video action labels, and prose are different target styles; short media outputs may reflect transfer and target-style calibration.
- Exact normalized reference overlap with the 1M training set is zero for 180 stored reference strings. This does not exclude paraphrases or upstream pretraining overlap.
- This is a quick three-architecture comparison with one seed and 20 items per modality. Cite it as exploratory. Claims about human caption quality need larger and independent evaluation.
