# Implementation Plan

## 目标

在这个 fork 基础上建立一套可持续维护的认知、验证和开发流程，为后续创建我们自己的 TTS 应用做准备。

---

## 阶段 1：完成仓库认知 ✅ 已完成

- 梳理技术栈 → `memory-bank/tech-stack.md`
- 梳理目录结构和文件职责 → `memory-bank/architecture.md`
- 建立进展记录文档 → `memory-bank/progress.md`

---

## 阶段 2：验证仓库可运行性 ✅ 已完成

### 已完成

- 配置本地 Python 虚拟环境 `.venv`（Python 3.13）
- 安装 CUDA 版 torch：`torch-2.11.0+cu128`，在 GTX 1660 上确认 `cuda.is_available() = True`
- 跑通 tokenizer 最小示例闭环（encode → decode → wav）
- 在 `--no-flash-attn --dtype float16` 配置下，通过 Gradio demo 验证了 1.7B CustomVoice 和 1.7B VoiceDesign 的完整 TTS 推理链路
- 音频主观收听确认正常

### 在此过程中发现并修复的问题

- **speech tokenizer 半精度 NaN bug**：原始代码将 speech tokenizer 与主模型一起加载成半精度，导致 codec decode 产生全 NaN waveform，音频表现为无声/纯噪音。修复方式：检测到半精度时强制将 speech tokenizer 保持为 float32，主模型精度不变。已在 `qwen_tts/core/models/modeling_qwen3_tts.py` 落地并通过实测验证。
- `qwen_tts/cli/demo.py` 增加了非有限值音频的显式错误报告。

### 阶段 2 的结论

- 这个 fork 现在能在本机 GPU 上正常跑起来。
- 运行最小闭环：加载模型 → 调用 `generate_custom_voice` / `generate_voice_design` → 收听输出。
- 最稳妥的主线组合：12Hz tokenizer + 1.7B CustomVoice 或 VoiceDesign，`float16`，`--no-flash-attn`。

---

## 阶段 3：确定 MVP 应用方向 ✅ 已完成

- 已决定采用四层本地 MVP 架构：Gradio GUI + FastAPI 服务层 + `subprocess` / 文件锁调度层 + 本地模型库 / SQLite / prompt 文件。
- 已决定默认围绕 12Hz tokenizer、离线本地模型路径和单 GPU 单飞行策略推进。
- 已明确首批闭环范围：模型登记、模型切换、TTS 推理、Base prompt 复用、单说话人训练任务调度。

---

## 阶段 4：建立 MVP 应用骨架 ✅ 已完成

- 已新增 `qwen_tts/app/runtime.py`，固定本地目录布局、默认推理参数、最小训练样本数和单 GPU 单飞行策略。
- 已新增 `qwen_tts/app/metadata.py`，完成 `app_data.db` 初始化、`models` / `voice_prompts` 建表，以及最小注册查询接口。
- 已新增 `qwen_tts/app/model_manager.py`，完成单例模型管理器、加载参数统一入口、模型复用 / 切换 / 卸载和推理互斥控制。
- 已新增 `qwen_tts/app/api/`，完成 FastAPI 基础骨架、健康检查、模型列表、音色列表和静态文件挂载。
- 已完成一次 Base 模型最小端到端验证，证明 `Voice Clone` 路径不再只停留在文档假设上。
- 已新增 `POST /api/v1/tts/generate` 与服务层分流逻辑，Phase 4 已进入实装中段。
- 已完成真实三类模型 HTTP 验证与一次真实模型切换显存观测，Phase 4 收口完成。
- 当前 MVP 应用骨架下一步进入 Phase 5 的音色提取与复用能力。

---

## 当前优先级

1. 进入 Phase 5，落地 Base 音色提取与 prompt 库复用。
2. 评估 0.6B 模型在本机的推理速度和显存占用。
3. 把当前 Gradio 入口逐步切成纯 HTTP 客户端，停止直接持有模型对象。

---

## 已知风险与约束

| 风险 | 状态 | 说明 |
|------|------|------|
| Transformers 版本锁 | 持续存在 | 固定在 4.57.3，升级可能破坏模型加载 |
| 示例脚本研究习惯 | 持续存在 | 不适合直接作为生产代码使用 |
| 微调仅覆盖单说话人 | 持续存在 | 多说话人微调需自行扩展 |
| GTX 1660 显存 6 GB | 持续存在 | 1.7B 模型可运行，更大模型需要评估 |
| SoX 需手动加入 PATH | 持续存在 | Windows 环境每次启动需 `$env:PATH = 'C:\Program Files (x86)\sox-14-4-2;' + $env:PATH` |
| flash-attn 未安装 | 持续存在 | Windows 不支持，使用 `--no-flash-attn` 绕过，性能略降 |
| speech tokenizer 半精度 NaN | **已修复** | 见阶段 2 修复说明 |
| 无 CUDA / CPU 环境限制 | **已解决** | CUDA torch 已安装，GPU 确认可用 |
