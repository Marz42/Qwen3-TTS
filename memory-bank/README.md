# Qwen3-TTS Fork Memory Bank

## 1. 目的

`memory-bank/` 是这个 fork 的维护文档区，用来沉淀本地已验证行为、MVP 实施状态、测试路径和运维注意事项。

适用读者：

- 维护这个仓库的开发者
- 需要复现实验和 MVP 功能的同事
- 需要判断“当前能做什么/不能做什么”的交付人员

## 2. 当前 MVP 状态

- MVP Phase 0-9 已全部完成实现与脚本化验收。
- Phase 9 清单脚本：`examples/test_phase9_e2e_checklist.py` 已通过。
- 当前验收重点是控制流与 API 行为可靠性；真实 GPU 训练吞吐仍受 `flash_attention_2` 环境限制。

## 3. MVP 四层结构

### 3.1 GUI 层（Gradio）

- 数据准备 GUI：`qwen_tts/cli/phase7_data_prep.py`
- 推理 GUI：`qwen_tts/cli/phase8_tts_gui.py`
- 原始 Demo：`qwen_tts/cli/demo.py`

### 3.2 API 与服务层（FastAPI）

- 应用入口：`qwen_tts/app/api/main.py`
- 路由：`qwen_tts/app/api/routes/`
- 服务：`qwen_tts/app/tts_service.py`、`qwen_tts/app/voice_service.py`

### 3.3 调度与任务层（subprocess + lock）

- 训练任务管理：`qwen_tts/app/job_manager.py`
- 锁与运行基线：`qwen_tts/app/runtime.py`

### 3.4 模型与音色库管理层（SQLite + 文件系统）

- 元数据：`qwen_tts/app/metadata.py`
- 模型仓库：`data/pretrained_models/`
- 音色 prompt：`data/prompts/`
- 任务目录：`data/jobs/`
- 推理输出：`static/outputs/`

## 4. 使用方式

### 4.1 启动 FastAPI

建议在仓库根目录执行：

```powershell
uvicorn qwen_tts.app.api.main:app --host 127.0.0.1 --port 8010
```

### 4.2 启动数据准备 GUI（Phase 7）

```powershell
python -m qwen_tts.cli.phase7_data_prep --api-base http://127.0.0.1:8010 --ip 127.0.0.1 --port 8020
```

### 4.3 启动推理 GUI（Phase 8）

```powershell
python -m qwen_tts.cli.phase8_tts_gui --api-base http://127.0.0.1:8010 --ip 127.0.0.1 --port 8030
```

### 4.4 运行脚本化验收

```powershell
python examples/test_phase6_job_manager.py
python examples/test_phase7_data_prep.py
python examples/test_phase8_tts_gui.py
python examples/test_phase9_e2e_checklist.py
```

## 5. 文档索引

### 5.1 活跃文档（Active）

- `progress.md`：项目当前状态、最近验证、风险与下一步
- `mvp-implementation-plan.md`：MVP 分阶段方案与验收清单
- `architecture.md`：仓库结构与模块职责
- `tech-stack.md`：技术栈与依赖边界
- `model-and-finetuning-guide.md`：模型调用和微调运行要点
- `gradio-demo-test-guide.md`：本机 Demo 与 GUI 测试手册

### 5.2 归档文档（Archived）

- `implementation-plan.md`：早期阶段性计划（已归档）
- `mvp-design.md`：早期架构草案（已归档）

> 归档文档保留用于追溯，不作为当前实施依据。

## 6. 已知约束

- `transformers==4.57.3` 视为锁定依赖。
- 12Hz tokenizer 是主线路径。
- 当前本机训练真实链路受 `flash_attention_2` 限制。
- 真实 ASR（SenseVoice-Small）尚未接入，Phase 7 当前为占位 ASR + 人工修订。

## 7. 文档维护规则

- 以“本地已验证行为”为准，不写猜测性结论。
- 若文档与当前代码不一致，优先更新文档并在 `progress.md` 记录。
- 过时内容必须标注“已归档”，避免混淆当前实施基线。
