# Qwen3-TTS Architecture

## 总览

这个仓库可以分成 5 个层次：

1. 根目录工程文件：定义包、安装方式、文档和许可证。
2. `qwen_tts/` 主包：核心推理能力、模型定义、tokenizer、CLI。
3. `examples/`：官方推理示例与基本测试脚本。
4. `finetuning/`：单说话人微调相关脚本。
5. `memory-bank/`：我们为维护 fork 额外建立的知识记录区。

## 根目录

### `LICENSE`

- Apache-2.0 许可证。
- 对 fork 和二次开发友好。

### `MANIFEST.in`

- 控制发布 Python 包时需要带上的附加资源。
- 对这类包含模型辅助资源的项目很重要。

### `pyproject.toml`

- 项目的 Python 打包入口。
- 定义包名 `qwen-tts`、版本、依赖、Python 版本范围。
- 声明 CLI 命令：`qwen-tts-demo`。
- 指定 setuptools 自动发现 `qwen_tts*` 包。

### `README.md`

- 项目的总入口文档。
- 介绍模型能力、安装方式、示例调用、Demo 启动、vLLM 使用和微调流程。
- 是理解仓库定位和官方推荐用法的第一优先级文档。

### `assets/`

- 根目录静态资源目录。
- 当前主要用于项目展示材料，而不是运行时代码。

### `memory-bank/`

- 当前 fork 的知识沉淀目录。
- 用于记录技术栈、架构、进展、计划等长期维护信息。

## `examples/`

这个目录的脚本不是单元测试框架，而是“可直接运行的能力验证脚本”。它们更像示例程序和手工验证入口。

### `examples/test_model_12hz_base.py`

- 演示 12Hz Base 模型的声音克隆能力。
- 覆盖单条和批量输入。
- 同时演示直接生成和先构造 `voice_clone_prompt` 再生成两种调用方式。

### `examples/test_model_12hz_custom_voice.py`

- 演示 CustomVoice 模型的受控语音生成。
- 输入包括文本、语言、说话人、可选指令。

### `examples/test_model_12hz_voice_design.py`

- 演示 VoiceDesign 模型根据自然语言描述生成目标声音。
- 适合验证风格控制与描述驱动合成。

### `examples/test_tokenizer_12hz.py`

- 演示 12Hz tokenizer 的 encode/decode。
- 覆盖字符串音频源、批量输入、dict 形式解码、numpy 输入等多种调用形式。

## `finetuning/`

这个目录是一套单说话人微调流水线，目标是基于 Base 模型训练出新的自定义说话人能力。

### `finetuning/README.md`

- 说明微调使用方式。
- 定义输入 JSONL 格式。
- 说明数据准备、训练命令和训练后推理验证。
- 明确当前只支持单说话人微调。

### `finetuning/prepare_data.py`

- 数据预处理脚本。
- 读取原始 JSONL，使用 `Qwen3TTSTokenizer` 提取训练音频的 `audio_codes`。
- 输出带 `audio_codes` 的新 JSONL，为 SFT 做准备。

### `finetuning/dataset.py`

- 定义 `TTSDataset`。
- 负责把文本、参考音频、离散音频码拼装成训练样本。
- 负责提取参考 mel，并构造训练时需要的各种 mask、label 和输入张量。

### `finetuning/sft_12hz.py`

- 单说话人 SFT 训练主脚本。
- 用 Accelerate 封装训练流程。
- 从 Base 模型加载初始化权重。
- 训练后写出 checkpoint，并把输出模型改写为 `custom_voice` 类型配置。
- 同时把目标说话人 embedding 写入权重中。

## `qwen_tts/`

这是项目主体包，外部用户主要通过这里提供的 API 和 CLI 使用仓库能力。

### `qwen_tts/__init__.py`

- 对外导出主 API：
  - `Qwen3TTSModel`
  - `VoiceClonePromptItem`
  - `Qwen3TTSTokenizer`
- 这是外部 `from qwen_tts import ...` 的主要入口。

### `qwen_tts/__main__.py`

- 包级入口。
- 直接执行 `python -m qwen_tts` 时只打印提示信息，引导用户使用 CLI 命令。

## `qwen_tts/cli/`

### `qwen_tts/cli/demo.py`

- Gradio Demo 的实现。
- 负责命令行参数解析、模型加载、UI 构建和事件绑定。
- 根据不同 checkpoint 自动判断模型类型，并切换对应交互界面。
- 是当前仓库离“应用”最近的一层。

## `qwen_tts/inference/`

这个目录提供最重要的推理封装，作用是把底层模型封装成更稳定、可复用的业务接口。

### `qwen_tts/inference/qwen3_tts_model.py`

- `Qwen3TTSModel` 的实现。
- 对 Hugging Face 的 `from_pretrained` 做一层业务封装。
- 负责：
  - 注册自定义 config/model/processor
  - 加载模型和 processor
  - 音频输入标准化
  - 语言/说话人校验
  - 语音克隆 prompt 构造
  - CustomVoice / VoiceDesign / VoiceClone 三种生成接口

### `qwen_tts/inference/qwen3_tts_tokenizer.py`

- `Qwen3TTSTokenizer` 的实现。
- 对 12Hz/25Hz tokenizer 做统一接口封装。
- 负责音频加载、重采样、编码、解码和输入输出格式兼容。

## `qwen_tts/core/`

这个目录是底层模型定义层，维护 fork 时通常在这里做最底层修改。

### `qwen_tts/core/__init__.py`

- 汇总导出 12Hz 和 25Hz tokenizer 的 config/model 类。
- 方便上层统一注册到 Hugging Face Auto 类体系中。

## `qwen_tts/core/models/`

这是主 TTS 模型定义目录。

### `qwen_tts/core/models/__init__.py`

- 导出主模型相关三类对象：
  - `Qwen3TTSConfig`
  - `Qwen3TTSForConditionalGeneration`
  - `Qwen3TTSProcessor`

### `qwen_tts/core/models/configuration_qwen3_tts.py`

- 定义主模型配置。
- 包含 speaker encoder、talker、code predictor 等多种子配置。
- 用于描述整个 TTS 生成模型的结构参数。

### `qwen_tts/core/models/modeling_qwen3_tts.py`

- 主 TTS 模型实现文件。
- 定义条件生成逻辑、说话人编码器、talker 相关模块、生成流程以及辅助函数。
- 这是仓库中最核心也最复杂的模型代码之一。

### `qwen_tts/core/models/processing_qwen3_tts.py`

- 文本 processor 实现。
- 基于 Qwen tokenizer 封装文本输入处理。
- 负责 padding、批处理和 chat template 兼容。

## `qwen_tts/core/tokenizer_12hz/`

这是当前主线语音 tokenizer 实现目录。

### `qwen_tts/core/tokenizer_12hz/configuration_qwen3_tts_tokenizer_v2.py`

- 12Hz tokenizer 的配置定义。
- 包含 encoder 与 decoder 的子配置。
- 说明输入采样率、输出采样率、量化器数量等关键超参数。

### `qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py`

- 12Hz tokenizer 的模型实现。
- 基于 `MimiModel` 和自定义 decoder 结构。
- 提供音频编码输出 `audio_codes`，以及由离散码恢复波形的逻辑。

## `qwen_tts/core/tokenizer_25hz/`

这是较旧一代 tokenizer 实现目录，目前更多是兼容和保留用途。

### `qwen_tts/core/tokenizer_25hz/configuration_qwen3_tts_tokenizer_v1.py`

- 25Hz tokenizer 的配置定义。
- 包括 encoder、DiT decoder、BigVGAN decoder 等配置。

### `qwen_tts/core/tokenizer_25hz/modeling_qwen3_tts_tokenizer_v1.py`

- 25Hz tokenizer 的主模型实现。
- 除离散码外，还包含 x-vector 和参考 mel 等额外条件信息。

### `qwen_tts/core/tokenizer_25hz/vq/`

- 25Hz tokenizer 使用的低层语音向量量化子模块。

#### `qwen_tts/core/tokenizer_25hz/vq/core_vq.py`

- 向量量化核心实现。
- 提供 codec 离散化过程中需要的基础 VQ 组件。

#### `qwen_tts/core/tokenizer_25hz/vq/speech_vq.py`

- 语音向量量化封装。
- 包含语音编码器量化逻辑和 x-vector 提取相关能力。

#### `qwen_tts/core/tokenizer_25hz/vq/whisper_encoder.py`

- 与 Whisper 风格编码器相关的声学特征提取与辅助逻辑。

#### `qwen_tts/core/tokenizer_25hz/vq/assets/`

- 25Hz tokenizer 的静态资源目录。

##### `qwen_tts/core/tokenizer_25hz/vq/assets/mel_filters.npz`

- mel filter bank 资源文件。
- 用于声学特征提取时的滤波器参数。

## 当前架构理解

从维护视角看，这个仓库有两条主线：

1. 推理主线：`qwen_tts/inference` + `qwen_tts/core` + `examples` + `cli/demo.py`
2. 微调主线：`finetuning` + `qwen_tts/core` + `qwen_tts/inference`

如果后续要做你们自己的应用，最可能复用的是：

- `Qwen3TTSModel`
- `Qwen3TTSTokenizer`
- `qwen_tts/cli/demo.py` 里的交互逻辑思路

最可能需要重构的是：

- 服务化接口层
- 配置管理
- 模型下载与缓存策略
- 错误处理与日志
- 微调脚本的工程化程度