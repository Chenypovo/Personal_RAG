部署与配置说明

背景

个人 RAG 项目通常需要在本地开发、演示环境和云端环境之间切换。部署设计需要让 embedding provider、LLM provider、索引路径和可选依赖都能通过配置控制。这样既能在本地离线跑评测，也能在有 API key 的环境中使用远端模型。

核心设计

配置集中在 .env 和 app/config.py。文本 embedding 支持 openai_compatible 和 local 两种 provider。openai_compatible 用于兼容远端 embedding API，local 用于 Hugging Face 模型。索引文件默认放在 data/index，上传文件默认放在 data/uploads，合成评测文档放在 data/uploads_eval。Streamlit 前端调用同一套 pipeline，因此命令行和网页端结果应保持一致。

参数与配置

默认本地 embedding 模型是 BAAI/bge-small-zh-v1.5，默认设备 EMBED_DEVICE=cpu。远端 embedding 默认读取 EMBED_BASE_URL、EMBED_API_KEY 和 EMBED_MODEL。LLM 默认读取 LLM_BASE_URL、LLM_API_KEY 和 LLM_MODEL。建库命令使用 scripts/build_index.py，检索评测命令使用 scripts/eval_retrieval.py。网页端默认通过 streamlit run web/streamlit_app.py 启动。

边界与失败处理

如果没有 API key，系统应选择 local embedding 或跳过需要远端模型的功能。如果本地模型尚未缓存，首次运行可能需要联网下载；离线环境应提前准备模型目录。索引路径不存在时，查询脚本会提示先建库。部署时不建议把 data/memory/memory.db 提交到公开仓库，因为它可能包含个人事实。生产环境应把上传目录和索引目录挂载为持久化卷。

结论

部署配置的重点是 provider 可切换、路径可复现、敏感数据不入库。只要评测命令和应用命令共用同一套配置，离线评测结果就能更好地解释线上问答表现。
