# 结构化块类型演示

## 列表块

检索链路的主要步骤：

- 文档接入与结构化切分
- 向量与 BM25 召回
- RRF 融合与 BGE 精排
- parent-child 父块回填

## 表格块

各后端检索对比：

| Backend | Recall@1 | MRR@4 |
| --- | ---: | ---: |
| 向量 | 0.31 | 0.54 |
| BM25 | 0.45 | 0.64 |
| 混合 | 0.40 | 0.66 |

## 代码块

构建索引命令：

```bash
python scripts/build_index.py --input-dir data/uploads --vector-store lancedb --embed-backend local
```

## 段落块

这一段是普通正文，用于演示 paragraph 类型的块，没有任何列表、表格或代码标记，只是连续的自然语言描述。
