# 模型调用与微调说明

## 适用范围

这份文档面向当前 fork 的本地维护和使用场景，重点解释四件事：

1. 三类模型 `CustomVoice`、`VoiceDesign`、`Base` 分别怎么调用。
2. `finetuning/` 这一套单说话人微调流程需要什么输入数据、数据格式是什么、训练参数怎么理解。
3. 训练好的 checkpoint 应该如何加载和推理。
4. `finetuning` 得到的模型与 `Base` 模型运行时声音克隆，在“提取/保存音色”这件事上的本质差异是什么。

更基础的公开示例请优先参考 [README.md](../README.md)；训练脚本原始说明请参考 [finetuning/README.md](../finetuning/README.md)。这份文档的重点不是重复它们，而是把当前代码里的真实行为和容易混淆的地方讲清楚。

---

## 统一加载方式

三类模型都通过同一个入口加载：

```python
from qwen_tts import Qwen3TTSModel

tts = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda:0",
    dtype=torch.float16,
    attn_implementation=None,
)
```

`Qwen3TTSModel.from_pretrained(...)` 本质上是对 Hugging Face `AutoModel.from_pretrained(...)` 和 `AutoProcessor.from_pretrained(...)` 的封装，所以常见的加载参数仍然是：

- `device_map`：例如 `cuda:0`、`cpu`
- `dtype` 或 `torch_dtype`：例如 `torch.float16`、`torch.bfloat16`
- `attn_implementation`：例如 `flash_attention_2` 或 `None`

这个入口会根据 checkpoint 里的 `tts_model_type` 判断当前模型属于哪一类，因此加载方式统一，但后续可用的生成函数不一样。

---

## 一、三个模型分别如何调用

### 总览

| 模型类型 | 典型 checkpoint | 主要调用函数 | 必填核心参数 | 是否需要参考音频 |
| --- | --- | --- | --- | --- |
| `CustomVoice` | `Qwen3-TTS-12Hz-1.7B-CustomVoice` / `0.6B-CustomVoice` | `generate_custom_voice(...)` | `text`、`speaker` | 否 |
| `VoiceDesign` | `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | `generate_voice_design(...)` | `text`、`instruct` | 否 |
| `Base` | `Qwen3-TTS-12Hz-1.7B-Base` / `0.6B-Base` | `generate_voice_clone(...)` | `text`，以及 `ref_audio` 或 `voice_clone_prompt` | 是 |

三类函数的返回值形式一致，都是：

```python
wavs, sr = ...
```

其中：

- `wavs` 是 `List[np.ndarray]`
- `sr` 是采样率整数

### 共同的生成参数

三类生成接口都支持一组相似的生成参数，常见的包括：

- `top_k`
- `top_p`
- `temperature`
- `repetition_penalty`
- `max_new_tokens`
- `subtalker_top_k`
- `subtalker_top_p`
- `subtalker_temperature`

这些参数最终会下传到底层 `generate(...)`，所以文档里看到的“采样相关参数”基本都可以按 Hugging Face 生成参数的理解来用。

### 1. CustomVoice

#### 调用函数

```python
generate_custom_voice(
    text,
    speaker,
    language=None,
    instruct=None,
    non_streaming_mode=True,
    **kwargs,
)
```

#### 必填参数

- `text`：要合成的文本。可以是单个字符串，也可以是字符串列表。
- `speaker`：说话人名字。必须是当前模型支持的 speaker 之一。

#### 常用可选参数

- `language`：语言名。可以传单个值，也可以传列表；不传时默认按 `Auto` 处理。
- `instruct`：控制语气、情绪、风格的自然语言指令。
- `non_streaming_mode`：当前更接近“非流式/模拟流式”的开关，不要把它理解成成熟的实时流式生成接口。

#### 典型调用

```python
wavs, sr = tts.generate_custom_voice(
    text="其实我真的有发现，我是一个特别善于观察别人情绪的人。",
    speaker="Vivian",
    language="Chinese",
    instruct="用平静、温柔、稍慢的语气说。",
)
```

#### 这一类模型的特点

- 不需要参考音频。
- 说话人是模型内部已经定义好的离散 speaker id。
- 支持批量输入：如果 `text` 是列表，而 `speaker`、`language`、`instruct` 只给了单个值，代码会自动广播到整批。

#### 需要特别注意

- `speaker` 会被校验；如果名字不在 `model.get_supported_speakers()` 里，会直接报错。
- 当前代码里，`0.6B-CustomVoice` 会把 `instruct` 直接置空，也就是**你传了也会被忽略**。这点应以实现为准，不要只看 README 的模型介绍。

### 2. VoiceDesign

#### 调用函数

```python
generate_voice_design(
    text,
    instruct,
    language=None,
    non_streaming_mode=True,
    **kwargs,
)
```

#### 必填参数

- `text`：要合成的文本。
- `instruct`：对声音风格、角色、情绪状态的自然语言描述。

#### 常用可选参数

- `language`：语言名；不传时默认 `Auto`。
- `non_streaming_mode`：同上。

#### 典型调用

```python
wavs, sr = tts.generate_voice_design(
    text="哥哥，你回来啦，人家等了你好久好久了，要抱抱！",
    language="Chinese",
    instruct="体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显。",
)
```

#### 这一类模型的特点

- 没有 `speaker` 参数，因为它不是从固定说话人列表里选人。
- 重点是让模型根据 `instruct` 设计一个声音角色。
- 也支持批量输入和参数广播。

#### 需要特别注意

- `instruct` 在签名上是必填参数。
- 但实现允许空字符串 `""`，这在效果上更接近“不给风格说明，直接生成默认声音”。

### 3. Base

`Base` 模型不是用 `speaker="某个名字"` 来调用，而是通过参考音频做声音克隆。

它有两种典型用法：

1. 直接把参考音频传给 `generate_voice_clone(...)`
2. 先用 `create_voice_clone_prompt(...)` 提前抽取音色信息，再重复复用

#### 3.1 直接克隆

```python
generate_voice_clone(
    text,
    language=None,
    ref_audio=None,
    ref_text=None,
    x_vector_only_mode=False,
    voice_clone_prompt=None,
    non_streaming_mode=False,
    **kwargs,
)
```

#### 必填核心逻辑

你至少要满足下面两种方案中的一种：

- 方案 A：传 `ref_audio`，并按需要传 `ref_text` / `x_vector_only_mode`
- 方案 B：传已经准备好的 `voice_clone_prompt`

#### 音频输入支持的类型

`ref_audio` 支持：

- 本地文件路径字符串
- URL
- base64 音频字符串
- `(numpy_array, sample_rate)` 二元组

#### 关于 `x_vector_only_mode`

- `x_vector_only_mode=True`：只使用说话人向量，不使用参考文本和参考音频码。
- `x_vector_only_mode=False`：进入 ICL 模式，会同时依赖参考文本和参考语音码。

#### 最重要的条件约束

如果 `x_vector_only_mode=False`，那么 `ref_text` 就是**必填**，因为模型要把参考文本和参考音频码一起当作条件。

#### 典型调用

```python
wavs, sr = tts.generate_voice_clone(
    text="She said she would be here by noon.",
    language="English",
    ref_audio="./clone.wav",
    ref_text="Okay. Yeah. I resent you. I love you.",
    x_vector_only_mode=False,
)
```

#### 3.2 先提取音色，再复用

```python
prompt = tts.create_voice_clone_prompt(
    ref_audio="./clone.wav",
    ref_text="Okay. Yeah. I resent you. I love you.",
    x_vector_only_mode=False,
)

wavs, sr = tts.generate_voice_clone(
    text="She said she would be here by noon.",
    language="English",
    voice_clone_prompt=prompt,
)
```

#### `create_voice_clone_prompt(...)` 做了什么

它会从参考音频中提取两类信息：

- `ref_spk_embedding`：说话人向量
- `ref_code`：参考音频对应的离散语音码

当 `x_vector_only_mode=True` 时，只保留说话人向量；
当 `x_vector_only_mode=False` 时，会同时保留 `ref_code`，这样生成时会更像“在参考音色和参考说话方式的上下文里继续说”。

#### 这一类模型的特点

- 没有 `speaker` 参数。
- 核心是“运行时”用参考音频临时抽取音色信息。
- 也支持批量输入；如果只有一个 prompt，但 `text` 是批量，代码会自动把这个 prompt 广播给整批文本。

---

## 二、如何进行 finetuning 微调

### 当前支持范围

当前 `finetuning/` 目录提供的是**单说话人**微调流程，针对的是：

- `Qwen3-TTS-12Hz-1.7B-Base`
- `Qwen3-TTS-12Hz-0.6B-Base`

它不是多说话人训练框架，也不是把 `VoiceDesign` 或官方 `CustomVoice` 继续训练的通用入口。现有脚本的目标，是把 `Base` 模型训练成“你自己的单说话人 CustomVoice”。

### 整体流程

训练路径可以理解成 3 步：

1. 准备原始 JSONL 数据。
2. 用 `prepare_data.py` 给每条样本补上 `audio_codes`。
3. 用 `sft_12hz.py` 做单说话人微调并导出 checkpoint。

对应的命令骨架如下：

```bash
python prepare_data.py \
    --device cuda:0 \
    --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
    --input_jsonl train_raw.jsonl \
    --output_jsonl train_with_codes.jsonl

python sft_12hz.py \
    --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --output_model_path output \
    --train_jsonl train_with_codes.jsonl \
    --batch_size 32 \
    --lr 2e-6 \
    --num_epochs 10 \
    --speaker_name speaker_test
```

如果你只是先冒烟验证流程，可以把第二条命令里的 `batch_size`、`lr`、`num_epochs` 改成脚本默认值那一组更保守的配置。

### 1. 原始输入数据长什么样

原始训练文件是 JSONL，也就是“一行一个 JSON 对象”。每一行至少包含 3 个字段：

- `audio`：目标训练音频路径
- `text`：这段训练音频的文字转写
- `ref_audio`：参考说话人音频路径

示例：

```jsonl
{"audio":"./data/utt0001.wav","text":"其实我真的有发现，我是一个特别善于观察别人情绪的人。","ref_audio":"./data/ref.wav"}
{"audio":"./data/utt0002.wav","text":"She said she would be here by noon.","ref_audio":"./data/ref.wav"}
```

### 2. 这些字段各自是什么意思

- `audio`：要让模型学会输出的真实目标音频。
- `text`：与 `audio` 对齐的转写文本。
- `ref_audio`：训练时拿来提取说话人表征的参考音频。

### 3. 数据格式要求和建议

#### 必须满足的要求

- 每条样本的 `audio` 和 `text` 必须真实对应。
- `ref_audio` 必须是同一个目标说话人的音频。
- `ref_audio` 在训练数据集读取时会走 mel 提取逻辑，当前实现里直接 `assert sr == 24000`，因此 **`ref_audio` 应该准备成 24kHz 音频**。

#### 强烈建议

- 对整个数据集使用**同一个** `ref_audio`。
- `audio` 也尽量统一成 24kHz、单声道 WAV，避免后续预处理和训练时出现额外不一致。
- 文本转写尽量干净，不要把明显的噪声标记、括号注释、与语音不一致的内容混进去。

### 4. 预处理阶段会新增什么内容

`prepare_data.py` 会调用 `Qwen3TTSTokenizer.encode(...)`，把每条 `audio` 变成离散语音码 `audio_codes`，然后输出新的 JSONL。

这个脚本内部按固定批大小分批编码，当前常量 `BATCH_INFER_NUM=32`。

也就是说，训练真正读入的不是原始 JSONL，而是“补了 `audio_codes` 的 JSONL”。

可以把结果理解成：

```jsonl
{
  "audio": "./data/utt0001.wav",
  "text": "...",
  "ref_audio": "./data/ref.wav",
  "audio_codes": [[...16个codebook值...], [...], ...]
}
```

其中：

- `audio_codes` 是二维离散码序列
- 在当前 12Hz tokenizer 路径下，每个时间步会有 16 路 codebook 值

### 5. 训练脚本实际做了什么

`sft_12hz.py` 的核心思路是：

1. 从 `ref_audio` 提取说话人 embedding。
2. 把这个 speaker embedding 放进 talker 的 codec embedding 位置中。
3. 用 `audio_codes` 监督模型学习目标语音码序列。
4. 训练结束后，把目标说话人的 embedding 固化进导出的 checkpoint 权重里。

这意味着这条微调链路的目标不是“每次生成时再去抽取一次音色”，而是把目标说话人的表示**写进模型**。

### 6. 训练参数怎么理解

这里有一个很容易误解的点：

- [finetuning/README.md](../finetuning/README.md) 给的是一组“推荐示例值”
- [finetuning/sft_12hz.py](../finetuning/sft_12hz.py) 里还有一组“脚本默认值”

它们**不是同一回事**。

| 参数 | README 推荐示例 | 脚本默认值 | 说明 |
| --- | --- | --- | --- |
| `batch_size` | `32` | `2` | README 更像“正式训练示例”，脚本默认值更保守，适合先跑通流程。 |
| `lr` | `2e-6` | `2e-5` | README 值更小，更偏稳妥；脚本默认学习率更激进。 |
| `num_epochs` | `10` | `3` | README 更像完整训练；脚本默认值更适合冒烟测试。 |
| `speaker_name` | `speaker_test` | `speaker_test` | 二者一致。 |

除此之外，训练脚本里还写死了几件事：

- `gradient_accumulation_steps=4`
- `mixed_precision="bf16"`
- `weight_decay=0.01`
- 梯度裁剪 `clip_grad_norm_=1.0`
- 模型加载时显式传了 `attn_implementation="flash_attention_2"`

### 7. 参数建议怎么选

如果你的目标是“先把流程跑通”，建议按更保守的思路来：

- 先从脚本默认 batch size 或更小值开始。
- 学习率优先参考 README 的 `2e-6`，除非你已经确认数据量和收敛行为允许更激进设置。
- epoch 可以先用 `3` 做冒烟，再决定是否升到 `10`。

如果你的目标是“更稳定的单说话人效果”，更值得优先保证的是：

- 数据质量
- 参考音频一致性
- 文本与语音对齐质量

而不是一上来追求更大的 batch 或更多 epoch。

### 8. 需要提前知道的现实限制

当前训练脚本在实现层面默认依赖：

- `bf16`
- `flash_attention_2`

如果你的本地训练硬件不支持这些配置，那么训练脚本本身还需要本地适配，不能把 README 的命令直接当成“任何机器都可以原样运行”。这和 demo 推理阶段可通过 `--no-flash-attn` 绕开的情况不完全一样。

---

## 三、如何调用训练好的 finetuning 模型

### 训练输出是什么

每个 epoch 结束后，脚本会在输出目录下写一个 checkpoint，例如：

- `output/checkpoint-epoch-0`
- `output/checkpoint-epoch-1`
- `output/checkpoint-epoch-2`

### 为什么训练后的模型要按 CustomVoice 来调用

`sft_12hz.py` 在保存 checkpoint 时做了两件关键事情：

1. 把 `config.json` 里的 `tts_model_type` 改成 `custom_voice`
2. 把 `speaker_name` 映射到内部 `spk_id=3000`

同时，它还会把训练得到的目标说话人 embedding 写入：

- `talker.model.codec_embedding.weight[3000]`

所以训练输出虽然来自 `Base`，但导出后的 checkpoint 在语义上已经被改造成“单说话人 CustomVoice 模型”。

### 加载方式

```python
tts = Qwen3TTSModel.from_pretrained(
    "output/checkpoint-epoch-2",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
```

### 推理方式

```python
wavs, sr = tts.generate_custom_voice(
    text="She said she would be here by noon.",
    speaker="speaker_test",
    language="English",
)
```

### 最重要的调用约束

- `speaker` 必须与你训练时传给 `--speaker_name` 的值一致。
- 训练后的模型不应该再按 `Base` 的思路去传 `ref_audio` / `ref_text` 调 `generate_voice_clone(...)`。
- 正确思路是：把它当成一个新的 `custom_voice` checkpoint，通过 `speaker_name` 直接选中目标说话人。

### 关于 `instruct` 的保守建议

训练后的 checkpoint 虽然会被改写成 `custom_voice` 类型，但这条微调链路本质上是从 `Base` 出发做单说话人适配。最保守、最稳妥的使用方法是：

- 先按 `generate_custom_voice(text=..., speaker=...)` 使用
- 不要默认认为它一定具备官方 1.7B `CustomVoice` checkpoint 那样已经充分验证过的 `instruct` 控制能力

如果你确实想在训练后的模型上叠加 `instruct`，建议单独做效果验证，而不是先入为主地当作“它天然就等价于官方 CustomVoice”。

---

## 四、finetuning 模型与 Base 声音克隆中的“提取/保存音色”有什么不同

这个问题最容易混淆，因为两者都和“音色”有关，但它们其实属于两条完全不同的路径。

### 1. Base 声音克隆的本质

`Base` 的声音克隆是：

- 在**运行时**读取参考音频
- 临时提取说话人 embedding
- 视模式决定是否再带上 `ref_code` 和 `ref_text`
- 然后立刻拿这些条件去生成目标文本

也就是说，它的音色不是事先学进模型里的，而是“每次调用时临时喂给模型”。

### 2. Base 的“保存音色”到底保存了什么

在 `Base` 路径下，如果你调用了 `create_voice_clone_prompt(...)`，或者在 Gradio demo 的“Save / Load Voice”页面里保存音色文件，本质上保存的是一份**运行时 prompt/cache**，里面通常包含：

- `ref_spk_embedding`
- 可能还有 `ref_code`
- `x_vector_only_mode`
- `icl_mode`
- 可选的 `ref_text`

这不是训练后的新模型，只是一份“以后可以重复复用的克隆条件”。

### 3. finetuning 模型的本质

finetuning 不是保存一份临时 prompt，而是：

- 用一批同一说话人的训练数据做优化
- 把目标说话人的表征固化进模型权重
- 导出一个新的 checkpoint

所以它不是“缓存一次音色提取结果”，而是“真的把模型改成更会说这个人的声音”。

### 4. 两者差异总表

| 维度 | Base 声音克隆 | finetuning 后的模型 |
| --- | --- | --- |
| 音色来源 | 运行时从 `ref_audio` 提取 | 训练后固化到模型权重 |
| 是否需要参考音频 | 需要，除非你已经有保存好的 prompt 文件 | 推理时不需要 |
| 是否需要参考文本 | `x_vector_only_mode=False` 时必须需要 | 推理时不需要 |
| 是否能保存音色 | 能，但保存的是 prompt/cache | 不是保存 prompt，而是导出整个 checkpoint |
| 调用入口 | `generate_voice_clone(...)` | `generate_custom_voice(...)` |
| 使用方式 | 每次生成都依赖外部参考条件 | 通过 `speaker_name` 直接调用 |
| 复现性 | 依赖参考音频和 prompt 内容 | 更稳定，调用路径更固定 |
| 成本 | 不用训练，但每次都要准备参考条件 | 前期要准备数据并训练 |

### 5. 更直白的理解方式

你可以把它们理解成：

- `Base` 克隆：像是“拿着一段样音，现场模仿这个人说一句话”
- `finetuning`：像是“专门训练出一个新的固定说话人槽位，以后直接点名这个人来读”

### 6. 什么时候该选哪条路

更适合用 `Base` 声音克隆的情况：

- 你只有少量参考音频
- 你想快速试一个人声克隆效果
- 你不想启动训练流程

更适合用 finetuning 的情况：

- 你已经有比较稳定的一批单说话人数据
- 你希望长期复用同一个新声音
- 你希望推理时不要每次再传 `ref_audio` / `ref_text`
- 你更看重一致性，而不是临时克隆的便利性

---

## 五、常见误区

### 误区 1：三个模型都只是参数不一样，函数随便换着用

不是。三类模型是通过 `tts_model_type` 区分的，加载后可调用的接口不同。`Base` 不支持 `generate_custom_voice(...)`，`CustomVoice` 也不支持 `generate_voice_clone(...)`。

### 误区 2：Base 也能直接传 `speaker="某个名字"`

不能。`Base` 的入口是参考音频克隆，而不是固定 speaker 选择。

### 误区 3：`ref_text` 永远都可以不传

不对。`Base` 模型里只要 `x_vector_only_mode=False`，`ref_text` 就是必填。

### 误区 4：微调后的模型还是应该按 Base 的方式使用

不对。当前脚本导出的 checkpoint 会被改写成 `custom_voice` 类型，所以训练后应该走 `generate_custom_voice(...)`。

### 误区 5：微调就是把 `create_voice_clone_prompt(...)` 的结果存起来

不对。保存 voice clone prompt 只是缓存运行时条件；finetuning 是修改模型权重，这两者不是一个层级的事情。

### 误区 6：README 里的训练参数就是脚本默认值

不对。README 给的是推荐示例值，脚本里还有另一组更保守的默认参数，文档和实际代码要区分看。

---

## 参考入口

- 公共模型与 API 示例： [README.md](../README.md)
- 微调流程原始说明： [finetuning/README.md](../finetuning/README.md)
- 推理接口真实实现： [qwen_tts/inference/qwen3_tts_model.py](../qwen_tts/inference/qwen3_tts_model.py)
- 微调训练脚本： [finetuning/sft_12hz.py](../finetuning/sft_12hz.py)
- 数据集读取逻辑： [finetuning/dataset.py](../finetuning/dataset.py)
- 当前 fork 的已知行为与限制： [memory-bank/progress.md](progress.md)