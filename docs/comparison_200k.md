# Fixed 80-item mixed evaluation

All outputs use greedy decoding with a 128-token limit. Each modality has 20 fixed items.

| Variant | Image ref cosine | Audio ref cosine | Video ref cosine | Media macro | Media tokens | Media EOS | Image correct−zero | Image correct−shuffled | Text val CE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1b | 0.705 | 0.698 | 0.709 | 0.704 | 7.8 | 100% | -1.598 | 1.662 | 2.182 |
| noise1_200k | 0.740 | 0.710 | 0.722 | 0.724 | 8.0 | 100% | -0.136 | 1.627 | 2.252 |
| prefix8_200k | 0.753 | 0.765 | 0.720 | 0.746 | 9.3 | 100% | 0.091 | 1.855 | 1.916 |

Media macro is an unweighted mean of image, audio, and video reference cosine. Cosine and within-modality retrieval reuse the frozen encoder and are diagnostics, not independent human judgments. Video references are action labels. Conditioning likelihood is teacher-forced and token-normalized; zero input is outside the encoder's unit-norm output manifold. The sample size is small, so inspect the full per-item JSONL and selected failures alongside the averages.
