# Qwen3-TTS Tech Stack

## 项目定位

这个项目是一个基于大语言模型和离散语音 tokenizer 的文本转语音仓库，目标领域包括：

- 文本转语音（TTS）
- 声音克隆（Voice Clone）
- 声音设计（Voice Design）
- 指令控制语音生成（Instruction-controlled speech generation）
- 语音离散编码与解码（Speech Tokenization / Codec）
- 单说话人微调（Single-speaker fine-tuning）
- 本地 Demo 与应用原型搭建

## 主要技术栈

### 1. 语言与工程基础

- Python：项目主体语言。
- setuptools + pyproject.toml：用于打包、安装和发布 `qwen-tts` Python 包。
- MANIFEST.in：控制发布包需要包含的非 Python 资源。

### 2. 深度学习与模型框架

- PyTorch：核心训练与推理框架。
- Transformers 4.57.3：项目最核心的模型加载、配置注册、Processor、AutoModel/AutoConfig 体系都依赖它。
- Accelerate：用于微调脚本中的多卡或混合精度训练封装。
- safetensors：用于保存微调后的模型权重。

这个仓库不是一个通用 Web 应用仓库，而是一个高度依赖 Hugging Face 生态的模型仓库。`Qwen3TTSModel` 和 `Qwen3TTSTokenizer` 都是围绕 Transformers 的 `from_pretrained` 风格封装出来的推理接口。

### 3. 语音处理相关库

- librosa：音频加载、重采样。
- torchaudio：音频相关依赖。
- soundfile：读写 wav 音频文件。
- sox：音频处理依赖。
- numpy：音频数组与中间张量转换。

### 4. 推理与模型能力

- Qwen3TTSModel：统一封装 3 类 TTS 能力。
  - Custom Voice
  - Voice Design
  - Voice Clone
- Qwen3TTSTokenizer：统一封装 12Hz 和 25Hz 两套语音 tokenizer 的 encode/decode。
- 12Hz tokenizer：当前主线 tokenizer，采用多 codebook 离散语音表示。
- 25Hz tokenizer：较旧一代 tokenizer，保留了 x-vector 和 mel 等附加条件信息。

### 5. 模型架构要点

- 主 TTS 模型：位于 `qwen_tts/core/models/`，包含配置、主模型和文本 processor。
- 文本侧：使用 Qwen 系列 tokenizer 与 chat template 风格输入格式。
- 说话人建模：包含 speaker encoder，用参考音频提取说话人表征。
- codec/token 侧：通过离散语音 token 进行生成与还原。
- 子说话器 / code predictor：用于预测多路 codec token。

从实现上看，这是一个“文本 token + 语音 codec token + 说话人条件”的联合生成系统，而不是传统的“文本前端 + 声学模型 + vocoder”松耦合三段式工程。

### 6. 应用与交互层

- Gradio：本地 Demo UI 所使用的界面框架。
- CLI：通过 `qwen-tts-demo` 命令启动 Demo。

这说明仓库已经具备基础产品化入口，但目前仍偏研究/模型演示导向，不是完整业务应用。

### 7. 模型与资源分发

- Hugging Face Hub：模型下载与 `from_pretrained` 加载主通道。
- ModelScope：国内镜像和替代下载源。
- README 中提供了两套模型下载方式。

### 8. 性能与硬件相关技术

- FlashAttention 2：README 和示例都推荐开启，用于降低显存占用和提升推理性能。
- bfloat16 / float16：主要建议的推理与训练数据类型。
- CUDA：示例默认按 GPU 环境编写。

## 技术栈结论

如果要维护这个 fork 并在此基础上做自己的应用，可以把它理解为三层：

1. 模型层：PyTorch + Transformers + 自定义 Qwen3 TTS 模型与 tokenizer。
2. 能力层：推理封装、语音编码解码、声音克隆、声音设计、微调脚本。
3. 应用层：CLI、Gradio Demo、未来你们自己的服务接口或业务产品。

对后续维护最关键的技术事实有三点：

- 仓库对 Transformers 版本耦合较深，升级依赖时需要非常谨慎。
- 12Hz tokenizer 是当前主线，应优先围绕它建设应用能力。
- 现有微调链路偏研究脚本风格，若要产品化，通常还需要补服务层、配置管理、日志、错误处理和部署流程。