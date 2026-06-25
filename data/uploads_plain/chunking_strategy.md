结构化 Chunking 策略

背景

个人知识库中的文档通常有标题、列表、代码块和表格。如果只按固定字符数切分，会把标题和正文分开，也可能把一个参数表拆到不同片段里，导致检索命中后上下文不完整。结构化 chunking 的目标是尽量保留 Markdown 层级，同时控制每个 chunk 的长度，使向量检索和 BM25 都能得到稳定输入。

核心设计

切分器先按一级和二级标题建立段落块，再在段落块内部按句子边界合并。每个 chunk 会继承最近的标题路径，例如“结构化 Chunking 策略 > 参数与配置”。代码块和表格被视为不可拆单元，除非单元本身超过硬限制。每个 chunk 的 metadata 包含 source、chunk_id、heading_path、start_char、end_char 和 text。chunk_id 在单个 source 内从 0 递增，用于生成 source#chunk_id 形式的 doc_id。

参数与配置

结构化 chunking 默认使用 450 token 作为 chunk_size，70 token 作为 overlap。遇到 Markdown 表格时，表头会复制到每个被拆分的表格片段中。单个代码块的硬上限是 900 token，超过后按空行切分。为了减少重复召回，同一文档相邻 chunk 的最大重叠比例限制为 25%。中文句子切分优先识别句号、问号、感叹号和分号。

边界与失败处理

如果文档没有标题，切分器会退化为普通滑窗切分，并把 heading_path 设为“untitled”。如果 PDF 解析出的文本行顺序混乱，系统会先按页码保留 page 字段，避免跨页拼接造成错误语义。对于极短文档，系统只生成一个 chunk，chunk_id 为 0。若某个 chunk 清洗后为空，会被跳过，但 chunk_id 不会回填，以保持索引和日志一致。

结论

结构化 chunking 通过标题继承、句子边界和重叠控制，提升了召回片段的可读性。它最适合测参数定位、章节定位和跨段综合问题，也是后续 chunk 级 qrels 的基础。
