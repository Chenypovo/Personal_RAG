BGE Reranker 精排设计

背景

初召回阶段通常追求高召回，会返回比最终上下文更多的候选。候选里可能有同一文档的重复片段，也可能有关键词命中但语义不相关的片段。Reranker 的职责是在候选数量较小的情况下，用交叉编码器重新判断 query 和 chunk 的相关性，把最可靠的证据排到前面。

核心设计

系统先从 vector、BM25 或 hybrid 中取 rerank_candidate_k 个候选，再把每个候选与原始 query 组成 pair 输入 BGE Reranker。Reranker 输出相关性分数后，系统按分数降序排序，并保留 rerank_top_k 条证据。精排后仍会执行 source 去重，避免最终上下文被同一文档占满。对于跨段综合问题，允许同一 source 保留最多 2 个 chunk。

参数与配置

BGE Reranker 默认使用 BAAI/bge-reranker-base。rerank_candidate_k 默认是 20，rerank_top_k 默认是 4。最大输入长度 max_length 设置为 512 token，超过后截断 chunk 文本而不是截断 query。Reranker 默认在 CPU 上运行，若检测到 CUDA 可用才切换到 GPU。精排开关名为 --rerank，关闭时系统直接使用初召回排序。

边界与失败处理

如果 transformers 或 torch 未安装，系统会懒加载失败并自动跳过精排，同时保留初召回结果。如果 reranker 模型下载失败，系统不会中断问答，而是记录 reranker_unavailable。对于非常短的精确参数问题，BM25 排名可能已经足够好，此时 Reranker 的收益可能有限。若候选数少于 rerank_top_k，则返回全部候选。

结论

Reranker 是提升 MRR@K 的关键模块，尤其适合把第 2 到第 4 名的相关证据推到第 1 名。它不替代召回，而是在召回后做更细粒度的相关性判断。
