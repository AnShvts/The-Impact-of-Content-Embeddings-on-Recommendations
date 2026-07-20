This project investigates the impact of multimodal content embeddings on recommendation quality in the context of video recommendation. Collaborative, content-based, and hybrid recommendation models were implemented and compared using the open-source VK-LSVD dataset.

Results

| Model | Recall@10 | Precision@10 | NDCG@10 |
|--------|-----------|--------------|---------|
| **Hybrid (cosine)** | **0.0057** | **0.0057** | **0.0125** |
| ALS | 0.0055 | 0.0055 | 0.0121 |
| Content-Only | 0.0055 | 0.0055 | 0.0055 |
| Popularity | 0.0040 | 0.0040 | 0.0029 |
| ItemKNN | 0.0005 | 0.0005 | 0.0011 |
