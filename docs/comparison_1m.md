# Fixed 80-item mixed evaluation

All outputs use greedy decoding with a 128-token limit. Each modality has 20 fixed items.

| Variant | Image ref cosine | Audio ref cosine | Video ref cosine | Media macro | Media tokens | Media EOS | Image correct−zero | Image correct−shuffled | Text val CE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_1m | 0.724 | 0.774 | 0.737 | 0.745 | 10.2 | 100% | 0.625 | 2.025 | 1.750 |
| noise1_1m | 0.754 | 0.763 | 0.722 | 0.747 | 9.1 | 100% | 1.274 | 2.094 | 1.760 |
| prefix8_1m | 0.793 | 0.791 | 0.727 | 0.770 | 11.9 | 98% | 3.284 | 3.766 | 1.647 |

Media macro is an unweighted mean of image, audio, and video reference cosine. Cosine and within-modality retrieval reuse the frozen encoder and are diagnostics, not independent human judgments. Video references are action labels. Conditioning likelihood is teacher-forced and token-normalized; zero input is outside the encoder's unit-norm output manifold. The sample size is small, so inspect the full per-item JSONL and selected failures alongside the averages.
