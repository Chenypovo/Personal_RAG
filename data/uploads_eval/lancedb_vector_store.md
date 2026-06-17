# LanceDB 与本地向量库设计

## 背景

向量库负责保存 embedding、文本片段和检索所需的 metadata。这个合成系统支持两种后端：轻量级 FAISS 本地索引和面向表结构管理的 LanceDB。FAISS 适合快速原型和纯向量召回，LanceDB 更适合需要过滤、版本字段和多模态记录统一存储的场景。

## 核心设计

LanceDB 表名默认是 personal_rag_chunks，每条记录包含 doc_id、source、chunk_id、modality、text、embedding、created_at 和 extra_json。doc_id 使用 source#chunk_id 生成，保证评测 qrels 可以稳定对齐。对于图片或视频帧，text 字段保存 OCR 或帧说明，modality 字段分别写 image 或 video_frame。extra_json 用于保存 page、heading_path、image_path、time_sec 等扩展字段。

## 参数与配置

LanceDB 默认向量列名为 embedding，向量维度由 embedding 模型第一次写入时确定。批量写入 batch_size 默认为 128。检索默认 top_k 为 8，带 metadata filter 时先过滤 source 或 modality，再做向量相似度排序。本地 FAISS 的索引文件默认保存到 data/index/faiss.index，元数据保存到 data/index/metadatas.json。LanceDB 数据目录默认是 data/lancedb。

## 边界与失败处理

当 LanceDB 不可用或依赖未安装时，系统会自动降级到 FAISSStore，不阻塞基础检索流程。如果写入时发现 embedding 维度与表结构不一致，系统会拒绝追加并提示重建索引。对于重复 doc_id，默认策略是覆盖旧记录，而不是插入重复行。若 metadata 中缺少 source，系统会使用 unknown_source，并在构建日志中标记为高优先级告警。

## 结论

LanceDB 方案强调结构化 metadata 与向量的统一管理，FAISS 方案强调最少依赖和速度。两者都必须保持 doc_id 规则一致，否则评测标注无法复用。
