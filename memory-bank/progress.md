# Project Progress

## 日期

- 最后更新：2026-05-09

## 当前阶段

- 已完成对 fork 项目的第一轮静态梳理。
- 已建立 `memory-bank` 作为后续认知沉淀区。
- 已明确本项目是一个以 Python + PyTorch + Transformers 为核心的 Qwen3-TTS 模型仓库，而不是完整业务应用。
- 已完成本地 Python 环境检查，并跑通最小 tokenizer 示例闭环。
- 已升级为 CUDA-enabled torch，在 GTX 1660 上完成完整 TTS 推理验证。
- 已定位并修复半精度下 speech tokenizer 产生全 NaN 波形的 bug，1.7B CustomVoice / VoiceDesign 音频现在可正常收听。
- 已启动本地 MVP 实施，并完成 Phase 0、Phase 1、Phase 2 与 Phase 3。
- 当前代码状态已具备运行基线模块、SQLite 元数据层、单例模型管理器、FastAPI 基础骨架、通用 TTS 接口、Base prompt 提取与复用、训练任务调度器。
- Phase 5 已完成验收，`extract_prompt` API、prompt 落盘、模拟重启后 `prompt_id` 复用与受控错误均已验证。
- Phase 6 已完成 TestClient 窄验证（8/8 通过），训练任务状态机、磁盘锁、启动恢复、并发拒绝、503 推理拦截均通过。

## 本次已完成内容

### MVP Phase 4 已完成验收

- 新增 `qwen_tts/app/runtime.py`，固定本地运行基线：
	- `data/pretrained_models/`
	- `data/prompts/`
	- `data/jobs/`
	- `static/outputs/`
	- `data/app_data.db`
- 新增 `qwen_tts/app/metadata.py`，完成 SQLite 元数据层。
- 新增 `qwen_tts/app/model_manager.py`，完成单例模型管理器。
- 新增 `qwen_tts/app/api/`，完成 FastAPI 基础壳层。
- 新增 `qwen_tts/app/tts_service.py` 与 `qwen_tts/app/api/routes/tts.py`，开始落地 Phase 4 通用生成接口。
- `qwen_tts/cli/demo.py` 已对齐 Phase 0 的单 GPU 单飞行策略：默认 `concurrency=1`，且显式拒绝大于 1 的并发值。
- 已完成数据库自动初始化逻辑，支持：
	- `models` 表的创建、注册、查询
	- `voice_prompts` 表的创建、注册、查询
- 已完成模型管理控制面逻辑，支持：
	- 同路径模型复用
	- 切换路径时卸载旧模型并重载
	- `gpu_lock` / 磁盘锁检查
	- 非阻塞推理互斥
	- 推理输出中的 `torch.Tensor` 递归转 CPU
- 已用本地 SQLite 往返验证：
	- 建表成功
	- 2 条模型记录写入并读回成功
	- 1 条 prompt 记录写入并读回成功
- 已用假加载器完成 Phase 2 窄验证：
	- 同路径重复加载只触发一次真实加载
	- 切换路径会触发重载
	- `gpu_lock=True` 时加载会返回受控错误
	- 并发推理时第二个请求会收到明确拒绝
- 已完成 Base 模型最小端到端验证：
	- 使用 `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
	- 在 `cuda:0 + float16 + no flash attention` 配置下成功生成 1 条语音
	- 输出文件为 `static/outputs/phase2_base_validation.wav`
	- 输出波形已验证全有限值
- 已完成 Phase 3 窄验证：
	- `/health` 返回 `200`
	- `/api/v1/models/list` 与 `/api/v1/voices/list` 返回结构化 JSON
	- `ValueError` 会被统一映射为 `400`
	- `/static/outputs/phase2_base_validation.wav` 可通过 FastAPI 静态挂载直接访问
- 已完成 Phase 4 首轮窄验证：
	- `POST /api/v1/tts/generate` 可按模型类型分流到 `custom_voice` / `voice_design` / `base`
	- `custom_voice` 缺 `speaker` 会返回 `400`
	- 生成结果可落盘并返回静态 URL，URL 可直接访问
	- 输出目录回收机制生效：阈值为 3 时，连续 4 次生成后目录文件数保持 3
- 已完成 Phase 4 真实模型 HTTP 验收：
	- `custom_voice` / `voice_design` / `base` 三类真实模型请求均返回 `200`
	- 真实验收序列为 `custom -> voice_design -> base -> custom`，用于覆盖切换路径
	- 四个生成结果 URL 均可访问并返回 `audio/wav`
- 已完成真实模型切换显存观测：
	- `before_custom_http`: allocated/reserved = `0 / 0 MB`
	- `after_custom_http`: allocated/reserved = `4316.84 / 4890 MB`
	- `after_voice_design_http`: allocated/reserved = `4316.84 / 4324 MB`
	- `after_base_http`: allocated/reserved = `4404.49 / 4424 MB`
	- `after_custom_again_http`: allocated/reserved = `4316.84 / 4324 MB`
	- 结论：切换后显存维持在稳定区间，未观察到随切换轮次线性累积。
- 验证过程中生成的占位模型目录、prompt 文件和临时数据库已清理，当前保留的是正式目录骨架和实现代码。

### MVP Phase 5 已完成验收
### MVP Phase 6 已完成 TestClient 窄验证

- 新增 `qwen_tts/app/job_manager.py`，实现训练任务状态机（`pending` / `running` / `succeeded` / `failed`）。
	- 任务状态存储在 `data/jobs/<job_id>/metadata.json`
	- 子进程日志落盘到 `data/jobs/<job_id>/prepare.log` 和 `data/jobs/<job_id>/train.log`
	- 磁盘 GPU 锁写入 `data/jobs/gpu.lock`（使用现有 `JobLockRecord` 格式）
	- 训练线程为 daemon 线程，不阻塞 API 主线程
	- 成功后自动在 `models` 表注册最终 checkpoint（`custom_voice` 类型）
- 新增 `POST /api/v1/models/train`：
	- preflight 校验：样本数 ≥ 5、模型类型必须为 `base`、batch_size 自动下调至 ≤ 样本数
	- 返回 `job_id` + `status=pending`（HTTP 202）
	- GPU 已忙时返回 `409`
- 新增 `GET /api/v1/jobs/{job_id}`：返回任务状态 + 最后 50 行日志（`log_tail`）
- 训练期间 TTS 接口返回 `503`（通过现有 `has_disk_gpu_lock()` + `GPUResourceBusyError` 机制实现）
- `create_app()` 启动时执行锁恢复：stale PID 清理脏 gpu.lock，关联 job 状态更新为 `failed`
- 已通过 8 项 TestClient 窄验证：
	1. 任务提交后进入 `running` 并最终到达 `succeeded`
	2. 训练锁存在时 TTS 返回 `503`
	3. 失败任务锁已释放，状态为 `failed`，error 有内容
	4. 样本数不足时返回 `400`（preflight 拒绝）
	5. 非 base 模型返回 `400`（preflight 拒绝）
	6. 启动锁恢复：stale lock + dead PID → lock 清理 + job 标记为 `failed`
	7. `GET /api/v1/jobs/{job_id}` 返回 `log_tail`
	8. 并发训练提交返回 `409`
- 注意：真实 `sft_12hz.py` 需要 `flash_attention_2`，本机不支持，当前验证使用 fake subprocess runner，真实训练需要支持 flash-attn 的环境。

- 新增 `qwen_tts/app/voice_service.py`，实现 Base prompt 提取与落盘注册。
- 新增 `POST /api/v1/voices/extract_prompt` 及对应请求/响应 schema。
- prompt 提取链路已支持：
	- Base 模型类型校验
	- `ref_audio`/`ref_text` 参数校验
	- prompt 序列化为纯 dict（仅含 Tensor + Python 基本类型），绕过 PyTorch >= 2.4 `weights_only=True` 限制
	- prompt 元数据写入 `voice_prompts` 表
- `POST /api/v1/tts/generate` 的 prompt 读取增加了 `weights_only=True` 与受控异常返回。
- 已通过真实 Base 模型 HTTP 验收（含模拟重启）：
	- `extract_prompt` 返回 `200`，`.pt` 文件真实落盘（19405 bytes）
	- `prompt_id` 复用生成返回 `200`，静态 URL 可访问 `audio/wav`
	- 模拟重启后（重新初始化 store + manager）同一 `prompt_id` 仍返回 `200`
	- 损坏 `.pt` 文件后生成返回受控错误 `400`

### 已阅读和确认的核心文件

- 根目录：`README.md`、`pyproject.toml`
- 包入口：`qwen_tts/__init__.py`、`qwen_tts/__main__.py`
- 推理层：`qwen_tts/inference/qwen3_tts_model.py`、`qwen_tts/inference/qwen3_tts_tokenizer.py`
- CLI：`qwen_tts/cli/demo.py`
- 主模型层：`qwen_tts/core/models/__init__.py`、`configuration_qwen3_tts.py`、`processing_qwen3_tts.py`
- tokenizer 层：12Hz 与 25Hz 的 config/model 文件
- 微调层：`finetuning/README.md`、`dataset.py`、`prepare_data.py`、`sft_12hz.py`
- 示例层：`examples/` 下 4 个脚本

### 已确认的仓库结论

- 当前主线 tokenizer 是 12Hz。
- 项目主要对外 API 是 `Qwen3TTSModel` 与 `Qwen3TTSTokenizer`。
- 当前本地应用入口是 Gradio Demo，而不是 HTTP 服务。
- 微调能力当前只覆盖单说话人场景。
- 仓库对 `transformers==4.57.3` 的耦合较深。
- 1.7B CustomVoice 和 1.7B VoiceDesign 支持自然语言指令控制。
- 0.6B CustomVoice 在代码里被显式关闭了 `instruct` 支持。
- Base / Voice Clone 接口不接收 `instruct` 参数，语气和语速控制更适合通过 CustomVoice 或 VoiceDesign 完成。

### 本地环境与运行验证

- 当前工作区自动化验证环境：`d:/Repos/Qwen3-TTS-Research/.venv/Scripts/python.exe`（Python 3.12.9）
- 安装了 CUDA 版 torch：`torch-2.11.0+cu128`、`torchaudio-2.11.0+cu128`
- GPU：NVIDIA GeForce GTX 1660，6 GB VRAM，driver 596.36，CUDA 13.2
- `torch.cuda.is_available()` 确认为 `True`
- SoX 已安装在 `C:\Program Files (x86)\sox-14-4-2\`（需手动加入 PATH）
- `flash-attn` 未安装；使用 `--no-flash-attn` 绕过，不影响功能
- Hugging Face 下载正常，但 Windows 上 symlink 降级警告仍存在（无害）

### 已确认的仓库结论

- 当前主线 tokenizer 是 12Hz。
- 项目主要对外 API 是 `Qwen3TTSModel` 与 `Qwen3TTSTokenizer`。
- 当前本地应用入口是 Gradio Demo，而不是 HTTP 服务。
- 微调能力当前只覆盖单说话人场景。
- 仓库对 `transformers==4.57.3` 的耦合较深。
- 1.7B CustomVoice 和 1.7B VoiceDesign 支持自然语言指令控制语气/语速/情感。
- 0.6B CustomVoice 在代码里被显式关闭了 `instruct` 支持。
- Base / Voice Clone 接口不接收 `instruct` 参数，情绪控制应通过 CustomVoice 或 VoiceDesign 完成。

### 已定位并修复的 Bug

**Speech tokenizer 半精度 NaN 问题**

- **现象**：使用 `--dtype float16` 或 `bfloat16` 时，Gradio demo 播放无声或纯噪音；控制台出现 `RuntimeWarning: invalid value encountered in cast`。
- **根因**：`Qwen3TTSForConditionalGeneration.from_pretrained` 原本把 speech tokenizer 也一起按半精度加载，codec 解码阶段数值溢出，导致全 NaN waveform。
- **修复文件**：`qwen_tts/core/models/modeling_qwen3_tts.py`
- **修复方式**：检测传入 dtype 是否为半精度，如果是则强制 speech tokenizer 以 `float32` 加载，主 TTS 模型仍保持半精度不变。
- **验证**：`CustomVoice float16` 和 `VoiceDesign float16` 均已通过实测，波形全有限值，音频可正常收听。
- **附加**：`qwen_tts/cli/demo.py` 中增加了非有限值音频的显式错误报告，避免以后再次无声失败。

## 当前产出

- `memory-bank/tech-stack.md`：技术栈与领域说明。
- `memory-bank/architecture.md`：目录与文件职责说明。
- `memory-bank/progress.md`：当前进展记录（本文件）。
- `memory-bank/implementation-plan.md`：实施计划（已从根目录移入）。
- `memory-bank/mvp-implementation-plan.md`：MVP 分阶段实施方案与状态对齐。
- `memory-bank/gradio-demo-test-guide.md`：Gradio demo 测试指南，含根因分析和已验证配置。
- `memory-bank/minimal_tokenizer_roundtrip.wav`：最小 tokenizer 闭环输出样本。
- `memory-bank/audition-samples/`：人工试听样本目录。
- `static/outputs/phase2_base_validation.wav`：Base 模型端到端验证输出样本。
- `qwen_tts/app/runtime.py`：MVP Phase 0 运行基线模块。
- `qwen_tts/app/metadata.py`：MVP Phase 1 SQLite 元数据层。
- `qwen_tts/app/model_manager.py`：MVP Phase 2 单例模型管理器。
- `qwen_tts/app/api/`：MVP Phase 3 FastAPI 基础骨架。
- `qwen_tts/app/tts_service.py`：MVP Phase 4 通用 TTS 服务与分流逻辑。
- `qwen_tts/app/voice_service.py`：MVP Phase 5 Base prompt 提取与复用。
- `qwen_tts/app/job_manager.py`：MVP Phase 6 训练任务状态机与磁盘锁管理。
- `qwen_tts/app/api/routes/jobs.py`：MVP Phase 6 `GET /api/v1/jobs/{job_id}` 路由。
- `examples/test_phase6_job_manager.py`：Phase 6 TestClient 窄验证脚本（8/8 通过）。

## 当前仍未完成的事项

- 尚未验证 0.6B 模型在本机的推理速度与显存占用。
- Phase 6 真实 sft_12hz.py 训练流程受 flash_attention_2 限制，当前验收仅覆盖状态机与控制流，未覆盖实际 GPU 训练。
- 尚未把 Gradio 从当前直连模型方式切到纯 HTTP 调 FastAPI（Phase 7/8）。

## 下一步建议方向

1. 进入 Phase 7：落地数据准备 GUI（Gradio 页面，HTTP → FastAPI）。
2. 进入 Phase 8：落地 TTS 推理 GUI。
3. 评估 0.6B 模型在本机的推理速度、显存占用和接口行为差异。