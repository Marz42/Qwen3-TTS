# Project Progress

## 日期

- 最后更新：2026-05-08

## 当前阶段

- 已完成对 fork 项目的第一轮静态梳理。
- 已建立 `memory-bank` 作为后续认知沉淀区。
- 已明确本项目是一个以 Python + PyTorch + Transformers 为核心的 Qwen3-TTS 模型仓库，而不是完整业务应用。
- 已完成本地 Python 环境检查，并跑通最小 tokenizer 示例闭环。
- 已升级为 CUDA-enabled torch，在 GTX 1660 上完成完整 TTS 推理验证。
- 已定位并修复半精度下 speech tokenizer 产生全 NaN 波形的 bug，1.7B CustomVoice / VoiceDesign 音频现在可正常收听。

## 本次已完成内容

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

- 当前工作区 Python 环境：`f:/Lab/Qwen3-TTS/.venv/Scripts/python.exe`（Python 3.13）
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
- `memory-bank/gradio-demo-test-guide.md`：Gradio demo 测试指南，含根因分析和已验证配置。
- `memory-bank/minimal_tokenizer_roundtrip.wav`：最小 tokenizer 闭环输出样本。
- `memory-bank/audition-samples/`：人工试听样本目录。

## 当前仍未完成的事项

- 尚未决定自己的应用形态（CLI 工具 / HTTP API / Web App / 桌面端）。
- 尚未梳理 fork 与上游之间的同步策略。
- 尚未验证 Base 模型（Voice Clone）端到端流程。
- 尚未验证 0.6B 模型在本机的推理速度与显存占用。

## 下一步建议方向

1. 确定自己的应用形态与首批功能边界（CustomVoice 优先）。
2. 基于 Gradio demo 骨架，搭建独立的应用入口层（参数配置、日志、错误处理与 demo 解耦）。
3. 评估 0.6B vs 1.7B 在 6 GB 显存下的实际运行指标，为选型提供数据支撑。
4. 决定是否以及如何维护与上游 fork 的同步策略。