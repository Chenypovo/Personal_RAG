# RAG 评测方法

## 背景

系统性评测需要把文档检索、记忆召回和生成忠实性拆开。若只观察最终回答，很难判断问题来自召回失败、排序失败还是生成幻觉。这个合成评测集先聚焦检索和记忆两类可量化任务，用 jsonl 保存 query 和 qrels，便于脚本重复运行。

## 核心设计

文档检索评测使用 queries.jsonl 和 qrels.jsonl。queries 每行包含 query_id、query 和可选 query_image。qrels 每行包含 query_id、doc_id 或 source/chunk_id，以及 relevance。记忆评测使用 memory_facts.jsonl、memory_queries.jsonl 和 memory_qrels.jsonl。文档 qrels 的最稳定粒度是 chunk 级 doc_id，格式为 source#chunk_id；在建索引前可以先标注 source 级别。

## 参数与配置

文档检索建议报告 Recall@1、Recall@4 和 MRR@4。记忆召回建议报告 Recall@1、Recall@3 和 MRR@3。每个 query 建议标注 1 到 3 个相关 chunk。小型 synthetic eval 的起步规模是 10 篇文档和 40 条查询。正式简历展示可以扩展到 30 到 80 条文档查询、50 到 100 条记忆事实和 100 条记忆查询。

## 边界与失败处理

如果某个 query 没有 qrels，评测脚本应该跳过并统计 n_skipped_no_qrels。如果 qrels 指向不存在的 doc_id，指标会被低估，因此建索引后必须用 metadatas.json 校验。对于跨段综合问题，qrels 可以包含多个相关 chunk，Recall@K 会按命中的相关集合比例计算。不要把不相关样例写入 qrels，relevance=1 表示相关即可。

## 结论

评测方法的关键是标注可复现。先用文档级 source 标注快速启动，再根据真实 chunk_id 生成 chunk 级 qrels，可以兼顾开发速度和指标可信度。
