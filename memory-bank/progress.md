# Project Progress

## 文档状态

- 状态：Active
- 最后更新：2026-05-08

## 当前总体状态

- MVP Phase 0-9 已全部完成。
- 核心验收脚本已通过：`examples/test_phase9_e2e_checklist.py`。
- 文档体系已收敛到“Active + Archived”双态管理。

## 阶段完成摘要

### Phase 0-3（基建层）

- 运行基线、目录规范、SQLite 元数据、ModelManager、FastAPI 壳层已稳定。

### Phase 4-6（能力层）

- 通用 TTS API、prompt 提取复用、训练调度状态机已稳定。
- 训练日志、锁恢复、并发拒绝等保护路径已验证。

### Phase 7-8（GUI 层）

- 数据准备 GUI 与推理 GUI 均已落地，均采用 HTTP 调用 API。

### Phase 9（验收层）

- 端到端路径 A/B、失败路径、回归项已脚本化并通过。

## 最新验证结论

### 已通过

- 三类模型推理路径
- Base prompt 提取与复用
- 训练任务提交、轮询、成功回流
- 训练失败释放锁
- 并发推理受控拒绝
- stale lock 启动恢复
- 输出目录清理策略

### 受环境限制

- 本机真实 `sft_12hz.py` GPU 训练仍受 `flash_attention_2` 限制。

## 当前产出

- 新增统一入口文档：`memory-bank/README.md`
- 新增 Phase 9 清单脚本：`examples/test_phase9_e2e_checklist.py`
- 完整更新 `memory-bank` 全部文档，并标注归档文档

## 下一步建议

1. 输出一份“人工验收记录”供交付归档。
2. 评估真实 ASR（SenseVoice-Small）接入收益。
3. 在可用硬件上补做真实训练链路验收。
4. 评估 0.6B 模型的速度/显存基准。
