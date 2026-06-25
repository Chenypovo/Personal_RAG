OCR 与多模态接入

背景

个人资料中常见图片、扫描 PDF、截图和短视频。多模态接入层负责把这些非纯文本资料转换成可检索记录。系统不假设所有环境都有完整 OCR 或视频处理依赖，因此必须具备可选增强和自动降级能力。

核心设计

文本文件直接进入文本 loader。PDF 优先提取内嵌文本，若页面文本为空再尝试 OCR。图片 loader 会保存 image_path，并把 OCR 结果写入 text 字段。视频 loader 按固定间隔抽帧，把每一帧保存为 image record，同时记录 time_sec。所有多模态记录都会统一成 entries 列表，每个 entry 包含 source、modality、text、chunk_id 和扩展 metadata。

参数与配置

图片 OCR 默认引擎是 rapidocr，语言模式为中英文混合。OCR 置信度阈值 ocr_min_confidence 设置为 0.45。视频抽帧间隔 frame_interval_sec 默认为 5 秒，最多抽取 60 帧。图片记录的 modality 写为 image，视频帧写为 video_frame。CLIP 多模态检索默认模型是 openai/clip-vit-base-patch32，设备默认使用 CPU。

边界与失败处理

当 OCR 失败时系统会降级：保留 image_path 和 modality，但 text 写为空字符串，并允许 CLIP 图像向量继续参与检索。如果 rapidocr 未安装，图片 loader 不会报错退出，而是记录 ocr_unavailable。若视频解码失败，系统会跳过该文件并在日志中写入 video_decode_failed。对于扫描 PDF，若 OCR 和文本提取都失败，该页不会生成文本 chunk。

结论

多模态接入层的核心价值是统一数据结构和可降级处理。即使 OCR 不可用，系统也能保留文件路径和模态信息，为后续图像检索或人工排查提供依据。
