# MVP 实施方案（Active）

## 文档状态

- 状态：Active
- 最后同步：2026-05-08
- 对齐代码范围：Phase 0-9

## 目标

构建本地可演示、可验收、可扩展的 Qwen3-TTS MVP：

1. GUI 层：Gradio
2. API 与服务层：FastAPI
3. 调度层：subprocess + 磁盘锁
4. 模型与音色管理层：SQLite + 本地文件

## 阶段总览

| 阶段 | 名称 | 状态 |
| --- | --- | --- |
| Phase 0 | 运行基线与目录布局 | 已完成 |
| Phase 1 | SQLite 元数据层 | 已完成 |
| Phase 2 | 单例 ModelManager | 已完成 |
| Phase 3 | FastAPI 基础壳层 | 已完成 |
| Phase 4 | 通用 TTS 服务 | 已完成 |
| Phase 5 | Base prompt 提取复用 | 已完成 |
| Phase 6 | 训练调度状态机 | 已完成 |
| Phase 7 | 数据准备 GUI | 已完成 |
| Phase 8 | 推理 GUI | 已完成 |
| Phase 9 | 端到端脚本化验收 | 已完成 |

## 已实现能力清单

### 1) 模型管理

- 模型登记、查询、切换与复用
- 单 GPU 单飞行推理互斥
- 训练/推理互斥

### 2) 推理能力

- 统一 `POST /api/v1/tts/generate`
- `custom_voice` / `voice_design` / `base` 分流
- 输出 WAV 落盘与 URL 返回

### 3) 音色库能力

- `POST /api/v1/voices/extract_prompt`
- `prompt_id` 复用生成
- prompt 可跨重启加载

### 4) 训练调度能力

- `POST /api/v1/models/train`
- `GET /api/v1/jobs/{job_id}`
- 任务状态机 + 日志落盘 + stale lock 恢复
- 训练成功自动回写模型仓库

### 5) GUI 能力

- Phase 7：样本收集、JSONL 生成、训练提交、状态查询
- Phase 8：模型选择、参数动态展示、生成试听/下载

## Phase 9 验收结论

脚本：`examples/test_phase9_e2e_checklist.py`

覆盖结果：

- 路径 A：推理闭环通过
- 路径 B：训练闭环通过
- 失败路径：参数错误、prompt 缺失、训练失败释放锁通过
- 回归项：并发拒绝、重启恢复、输出清理、日志追踪通过

## 已知限制（不作为当前功能缺失）

1. 真实 GPU 训练仍受 `flash_attention_2` 环境约束。
2. 真实 ASR 模型尚未接入（当前为占位 ASR + 人工修订）。
3. 0.6B 模型性能基准尚未专项评测。

## 交付建议

- 在当前基线下进入“真实训练环境验收”分支。
- 根据交付优先级决定是否接入 SenseVoice-Small。
- 若进入产品化阶段，补充 CI 和部署流水线文档。
