# Gradio Demo and GUI Test Guide (Fork Local)

## 1. 适用范围

本手册覆盖三类入口：

- 原始 Demo：`qwen_tts.cli.demo`
- Phase 7 数据准备 GUI：`qwen_tts.cli.phase7_data_prep`
- Phase 8 推理 GUI：`qwen_tts.cli.phase8_tts_gui`

## 2. 推荐本机推理参数

- `--device cuda:0`
- `--dtype float16`
- `--no-flash-attn`

原因：当前机器环境已验证该组合稳定，且能规避 flash-attn 依赖问题。

## 3. 启动命令

### 3.1 原始 Demo

```powershell
python -m qwen_tts.cli.demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --device cuda:0 --dtype float16 --no-flash-attn --ip 127.0.0.1 --port 8000
```

### 3.2 FastAPI

```powershell
uvicorn qwen_tts.app.api.main:app --host 127.0.0.1 --port 8010
```

### 3.3 Phase 7 GUI

```powershell
python -m qwen_tts.cli.phase7_data_prep --api-base http://127.0.0.1:8010 --ip 127.0.0.1 --port 8020
```

### 3.4 Phase 8 GUI

```powershell
python -m qwen_tts.cli.phase8_tts_gui --api-base http://127.0.0.1:8010 --ip 127.0.0.1 --port 8030
```

## 4. 功能检查点

### 4.1 Phase 7

- 样本收集（音频/zip）
- 样本表编辑（`audio/text/asr_text`）
- 生成 train jsonl
- 训练提交与 job 查询

### 4.2 Phase 8

- 拉取模型列表与 prompt 列表
- 按模型类型动态参数
- 生成音频并播放
- 训练占用时显示 `503`

## 5. 失败排查

### 5.1 503 错误

通常表示训练锁存在或推理并发冲突，先检查：

- `data/jobs/gpu.lock` 是否存在
- 是否有正在运行的训练任务

### 5.2 prompt 相关错误

- `prompt_id` 记录存在但文件缺失会返回 404
- prompt 文件损坏会返回 400

### 5.3 音频无效

- 优先检查输入音频路径是否有效
- 保证文本字段非空

## 6. 推荐自动化验证

```powershell
python examples/test_phase7_data_prep.py
python examples/test_phase8_tts_gui.py
python examples/test_phase9_e2e_checklist.py
```
