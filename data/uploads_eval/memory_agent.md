# 长期记忆 Agent 设计

## 背景

长期记忆用于保存用户的稳定事实、偏好、约束和阶段性目标。它不同于文档库：文档库主要来自上传资料，长期记忆主要来自对话中抽取的个人信息。记忆模块需要支持新增、更新、冲突检测和召回评测，避免把过时偏好误用于当前回答。

## 核心设计

每条记忆由 fact_id、fact_object、fact_content、visibility、state、source 和 updated_at 组成。fact_object 的推荐类别包括 school、work、project、preference、skill、location、constraint 和 goal。抽取器从用户消息中识别候选事实，合并器再判断它是新事实、补充事实还是与旧事实冲突。检索时使用记忆向量索引召回 top_k 条，再按 state 和 visibility 过滤。

## 参数与配置

记忆召回默认 top_k 为 5，评测建议使用 Recall@1、Recall@3 和 MRR@3。相似事实合并阈值 memory_merge_threshold 设置为 0.78。冲突检测阈值 memory_conflict_threshold 设置为 0.72。ACTIVE 状态的事实才会进入检索，SUPERSEDED 状态只保留审计用途。默认 visibility 是 PUBLIC，敏感事实可以标记为 PRIVATE 并禁止进入普通回答上下文。

## 边界与失败处理

当新事实和旧事实冲突时，系统不会直接删除旧事实，而是把旧事实标记为 SUPERSEDED，并把新事实写为 ACTIVE。如果抽取器置信度低于 0.6，候选事实会进入待确认队列，不直接写入记忆库。对于临时情绪和一次性指令，系统不写入长期记忆。若记忆检索为空，回答仍可继续使用文档证据。

## 结论

长期记忆 Agent 的重点是稳定、可撤销和可评测。通过 fact_id 级 qrels，可以单独评估记忆召回是否能处理改写问法和相近事实干扰。
