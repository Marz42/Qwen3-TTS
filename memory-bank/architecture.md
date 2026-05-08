# Qwen3-TTS Architecture (Fork Local)

## 1. 仓库分层

1. 根目录工程层：`pyproject.toml`、`README.md`、许可证与依赖定义。
2. 核心包层：`qwen_tts/`（推理、模型、tokenizer、CLI、MVP app）。
3. 微调脚本层：`finetuning/`（单说话人训练链路）。
4. 示例与验收层：`examples/`（Phase 6-9 脚本化验证）。
5. 运行资源层：`data/` 与 `static/`。
6. 维护文档层：`memory-bank/`。

## 2. MVP 代码结构

### 2.1 运行基线

- `qwen_tts/app/runtime.py`
- 负责目录布局、默认推理参数、训练样本下限、输出清理策略。

### 2.2 元数据层

- `qwen_tts/app/metadata.py`
- SQLite 两张表：`models`、`voice_prompts`。

### 2.3 模型管理层

- `qwen_tts/app/model_manager.py`
- 单例模型管理、模型切换、推理互斥、GPU busy 检查。

### 2.4 服务层

- `qwen_tts/app/tts_service.py`
- `qwen_tts/app/voice_service.py`
- 统一分发 `custom_voice` / `voice_design` / `base` 调用。

### 2.5 调度层

- `qwen_tts/app/job_manager.py`
- 训练任务状态机、日志落盘、锁恢复、模型回流注册。

### 2.6 API 层

- `qwen_tts/app/api/main.py`
- `qwen_tts/app/api/routes/`
- 路由：`models`、`voices`、`tts`、`jobs`、`data_prep`。

### 2.7 GUI 层

- `qwen_tts/cli/phase7_data_prep.py`
- `qwen_tts/cli/phase8_tts_gui.py`
- 原始 demo：`qwen_tts/cli/demo.py`

## 3. 数据与文件布局

- `data/app_data.db`：SQLite 元数据
- `data/pretrained_models/`：本地模型目录
- `data/prompts/`：prompt 文件
- `data/jobs/`：任务目录与 `gpu.lock`
- `data/datasets/`：训练数据与导入目录
- `static/outputs/`：推理输出 wav

## 4. 关键控制点

- 单 GPU 单飞行：训练与推理互斥。
- 训练锁：磁盘锁优先，避免重启状态丢失。
- Prompt 可移植性：保存时 CPU 化，读取用 `weights_only=True` + 重建。
- 输出清理：按阈值删除最旧 wav，防止目录膨胀。

## 5. 当前架构状态

- Phase 0-9 已按方案实现并完成脚本化验收。
- Phase 9 脚本：`examples/test_phase9_e2e_checklist.py`。
