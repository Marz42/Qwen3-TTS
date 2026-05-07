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

## 阶段 3：定义我们的应用方向 🔲 待决策

- 明确目标产品形态：脚本工具、CLI、HTTP API、Web App、桌面应用。
- 明确首个必须支持的能力：
  - Custom Voice（固定说话人 + 指令控制）
  - Voice Design（纯文字描述声音风格）
  - Voice Clone（参考音频克隆）
  - Tokenizer encode/decode（独立使用场景）
- 明确输入输出接口形式（同步/异步、批量还是单条）。
- 明确是否需要保留微调能力作为应用的一部分。

---

## 阶段 4：建立我们自己的应用骨架 🔲 待开始

- 决定是否复用 `qwen_tts/cli/demo.py` 作为原型基础，还是另立入口。
- 如需服务化，建立独立的服务入口层，与模型加载层解耦。
- 把模型加载、参数配置、日志和错误处理从示例脚本中分离。
- 为 fork 自己的代码增加配置文件、运行说明和最小测试。

---

## 当前优先级

1. 确定应用形态（阶段 3 的第一个决策点）。
2. 评估 0.6B vs 1.7B 在 6 GB 显存下的实际运行指标，为选型提供数据支撑。
3. 基于上述决策启动阶段 4。

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
