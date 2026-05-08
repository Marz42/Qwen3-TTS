# Qwen3-TTS Tech Stack (Fork Local)

## 1. 语言与工程基础

- Python 3.12.x（当前本地已验证）
- 打包：`setuptools` + `pyproject.toml`
- 路径与文件：`pathlib`
- 本地数据库：`sqlite3`
- 并发控制：`threading`
- 子进程调度：`subprocess`

## 2. AI 与训练推理栈

- `torch` / `torchaudio`
- `transformers==4.57.3`（锁定）
- `accelerate`
- `safetensors`

## 3. 音频处理栈

- `librosa`
- `soundfile`
- `sox`
- `numpy`

## 4. 服务与界面栈

- FastAPI（MVP API 层）
- Gradio（Phase 7/8 GUI 层）
- Starlette StaticFiles（输出音频静态访问）

## 5. 项目主能力对象

- `Qwen3TTSModel`
- `Qwen3TTSTokenizer`

## 6. 架构性技术事实

- 12Hz tokenizer 是当前主线。
- 训练与推理统一受单 GPU 单飞行策略约束。
- 模型加载由 `ModelManager` 统一管控，不走分散式加载。
- 任务状态机与磁盘锁由 `JobManager` 管控。

## 7. 当前限制

- 本机未安装 `flash-attn`，真实训练吞吐不在当前验收范围。
- 真实 ASR（SenseVoice-Small）未接入，Phase 7 采用占位 ASR + 人工修订。
- `transformers` 版本不建议在 MVP 收口后随意升级。

## 8. 依赖变更原则

- 先在 `examples/test_phase6_job_manager.py`、`examples/test_phase7_data_prep.py`、`examples/test_phase8_tts_gui.py`、`examples/test_phase9_e2e_checklist.py` 验证。
- 任何升级若影响 `from_pretrained` 或 tokenizer decode 路径，视为高风险变更。
