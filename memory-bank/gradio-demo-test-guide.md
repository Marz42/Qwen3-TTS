# Gradio Demo Manual Test Guide

## 结论先说

你这台机器手动跑 Gradio demo 时，第一优先级配置应该是：

- `--device cuda:0`
- `--dtype float16`
- `--no-flash-attn`

不要直接用 demo 默认值，因为 `qwen_tts/cli/demo.py` 的默认 dtype 是 `bfloat16`，而你的显卡是 GTX 1660，这类卡更适合先用 `float16`。如果错误地用 `bfloat16`，最常见的结果不是直接报错，而是生成质量异常，甚至只剩噪音。

## 已确认的根因

现在已经确认，问题的根因不是“1660 完全不能跑半精度 TTS”，而是这个项目原本会把 `speech_tokenizer` 也跟着主模型一起加载成 `float16` 或 `bfloat16`。

在这份项目里，真正把离散语音 token 还原成 waveform 的是 `speech_tokenizer.decode(...)`。实测表明：

- 主 TTS 模型用半精度生成 token 可以工作。
- 但 `speech_tokenizer` 一旦也用半精度做 decode，就可能直接生成全 `NaN` waveform。
- Gradio 再把这些 `NaN` waveform 转成 `int16` 时，就会出现你看到的：

```text
RuntimeWarning: invalid value encountered in cast
```

也就是说：

- 你看到的“纯噪音”，本质上不是正常音频，而是无效 waveform 在 UI 层被错误转换后的结果。
- `CustomVoice` 和 `VoiceDesign` 都可能受这个问题影响，只是症状不一定每次都完全一致。

当前仓库已经修正为：即便主模型用 `float16`，`speech_tokenizer` 也会强制用 `float32` 加载，以避免这个数值稳定性问题。

## Gradio Demo 实际怎么调用模型

### 启动参数

Gradio demo 的入口在 `qwen_tts/cli/demo.py`。

它的 `main()` 最后会这样加载模型：

```python
tts = Qwen3TTSModel.from_pretrained(
    ckpt,
    device_map=args.device,
    dtype=dtype,
    attn_implementation=attn_impl,
)
```

其中：

- `--device` 默认是 `cuda:0`
- `--dtype` 默认是 `bfloat16`
- `--flash-attn` 默认开启
- `--no-flash-attn` 时，传入的 `attn_implementation` 会变成 `None`

结合目前官方 1.7B checkpoint 的 `config.json`，`_attn_implementation` 也是 `None`，所以在你这边 `--no-flash-attn` 的效果是成立的，不会偷偷回落到 `flash_attention_2`。

### CustomVoice 页面的调用

CustomVoice 页面点击 `Generate` 后，实际调用的是：

```python
tts.generate_custom_voice(
    text=text.strip(),
    language=language,
    speaker=speaker,
    instruct=(instruct or '').strip() or None,
    **kwargs,
)
```

也就是说：

- 文本框内容就是 `text`
- 语言下拉框就是 `language`
- 说话人下拉框就是 `speaker`
- 指令输入框就是 `instruct`

这是最适合先验证“情感、语气、语速指令”的页面。

### VoiceDesign 页面的调用

VoiceDesign 页面点击 `Generate` 后，实际调用的是：

```python
tts.generate_voice_design(
    text=text.strip(),
    language=language,
    instruct=design.strip(),
    **kwargs,
)
```

这里的 `instruct` 是对声音风格和情感状态的自然语言描述，不是固定 speaker 选择。

### Base 页面的调用

Base 页面点击 `Generate` 后，实际调用的是：

```python
tts.generate_voice_clone(
    text=text.strip(),
    language=language,
    ref_audio=at,
    ref_text=(ref_txt.strip() if ref_txt else None),
    x_vector_only_mode=bool(use_xvec),
    **kwargs,
)
```

Base 主要是声音克隆，不是最适合先验证“情绪控制”的页面。

## 推荐的手动测试顺序

### 第一轮：先测 CustomVoice

原因：

- 有固定 speaker，不需要参考音频。
- 最容易区分“无指令”和“有情绪/语速指令”的差异。
- 排查问题更简单。

### 第二轮：再测 VoiceDesign

原因：

- 它更偏“描述一个声音角色”。
- 如果你已经确认 CustomVoice 正常出声，再看 VoiceDesign 的风格控制更容易判断效果。

### 第三轮：最后才测 Base

原因：

- Base 需要参考音频和参考文本。
- 变量更多，更不适合作为第一轮排障入口。

## 手动启动命令

下面的命令建议在仓库根目录执行。

### 1. 激活环境并注入 SoX

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& f:\Lab\Qwen3-TTS\.venv\Scripts\Activate.ps1
$env:PATH = 'C:\Program Files (x86)\sox-14-4-2;' + $env:PATH
```

### 2. 确认 CUDA 版 torch 已生效

```powershell
f:\Lab\Qwen3-TTS\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

你期待看到：

- `torch` 带 `+cu128`
- `torch.version.cuda` 不是 `None`
- `torch.cuda.is_available()` 是 `True`

### 3. 启动 1.7B CustomVoice demo

为了确保用的是当前仓库代码，建议直接用模块方式启动：

```powershell
$env:PATH = 'C:\Program Files (x86)\sox-14-4-2;' + $env:PATH
f:\Lab\Qwen3-TTS\.venv\Scripts\python.exe -m qwen_tts.cli.demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --device cuda:0 --dtype float16 --no-flash-attn --ip 127.0.0.1 --port 8000
```

浏览器打开：

```text
http://127.0.0.1:8000
```

### 4. 启动 1.7B VoiceDesign demo

```powershell
$env:PATH = 'C:\Program Files (x86)\sox-14-4-2;' + $env:PATH
f:\Lab\Qwen3-TTS\.venv\Scripts\python.exe -m qwen_tts.cli.demo Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --device cuda:0 --dtype float16 --no-flash-attn --ip 127.0.0.1 --port 8001
```

浏览器打开：

```text
http://127.0.0.1:8001
```

### 5. 如果你要测 Base

```powershell
$env:PATH = 'C:\Program Files (x86)\sox-14-4-2;' + $env:PATH
f:\Lab\Qwen3-TTS\.venv\Scripts\python.exe -m qwen_tts.cli.demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --device cuda:0 --dtype float16 --no-flash-attn --ip 127.0.0.1 --port 8002
```

## CustomVoice 手动测试指南

### 页面字段怎么填

- `Text`：先用短句。
- `Language`：明确指定，不要先用 `Auto`。
- `Speaker`：先选该语言的原生 speaker。
- `Instruction`：从空白开始，再逐步加风格控制。

### 推荐的第一组测试

文本：

```text
其实我真的有发现，我是一个特别善于观察别人情绪的人。
```

语言：

```text
Chinese
```

speaker：

```text
Vivian
```

按下面顺序各跑一遍：

1. 空指令
2. `用平静、温柔、稍慢的语气说。`
3. `用特别愤怒、语速快、咬字更重的语气说。`
4. `用很悲伤、语速慢、像快哭出来一样的语气说。`
5. `用兴奋、轻快、语速更快的语气说。`

你要关注的不是“音色是否变化很多”，而是：

- 能不能稳定发声
- 发音是否清楚
- 语速是否明显不同
- 情绪是否有主观可辨差异

### 英文测试建议

文本：

```text
She said she would be here by noon.
```

语言：

```text
English
```

speaker：

```text
Ryan
```

指令顺序：

1. 空指令
2. `Speak calmly and gently, with a slightly slower pace.`
3. `Speak in a very angry tone, fast and forceful.`
4. `Speak in a sad tone, slowly, as if you are about to cry.`

## VoiceDesign 手动测试指南

### 页面字段怎么填

- `Text`：依然先用短句。
- `Language`：明确指定。
- `Voice Design Instruction`：写完整自然语言描述。

### 推荐的中文测试

文本：

```text
哥哥，你回来啦，人家等了你好久好久了，要抱抱！
```

语言：

```text
Chinese
```

描述可以试这几种：

1. `年轻女声，温柔、轻快，语速稍快，带明显开心情绪。`
2. `年轻女声，委屈、撒娇，音调偏高，语速稍慢。`
3. `年轻女声，冷静克制，语速偏慢，情绪压低但清晰。`

### 推荐的英文测试

文本：

```text
It's in the top drawer... wait, it's empty? No way, that's impossible!
```

语言：

```text
English
```

描述可以试：

1. `Speak in an incredulous tone with a hint of panic.`
2. `Speak calmly, slowly, and gently, as if reassuring someone.`
3. `Speak quickly, sharply, and angrily, with strong emphasis.`

## 为什么你之前可能只听到噪音

最值得优先怀疑的是这几个因素：

### 1. 用了 demo 默认的 `bfloat16`

`qwen_tts/cli/demo.py` 默认 dtype 是 `bfloat16`。

对于你的 GTX 1660，这不是首选。手动测试时应明确改成：

```text
--dtype float16
```

### 2. 第一次就测了 Base 或者长文本

Base 变量更多，长文本也更容易把问题放大。先用 CustomVoice + 单句短文本排查最稳。

### 3. 浏览器里听到的是坏结果，但模型实际输出不一定坏

为排除 UI 层问题，可以再做一次“非 UI 保存 wav”对照测试。

## UI 之外的最小对照测试

如果你想确认到底是模型本身有问题，还是 Gradio 层播放有问题，可以直接跑下面的短脚本。

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice',
    device_map='cuda:0',
    dtype=torch.float16,
)

wavs, sr = model.generate_custom_voice(
    text='其实我真的有发现，我是一个特别善于观察别人情绪的人。',
    language='Chinese',
    speaker='Vivian',
    instruct='用平静、温柔、稍慢的语气说。',
    max_new_tokens=1024,
)

sf.write('custom_voice_manual_test.wav', wavs[0], sr)
print('saved', sr, len(wavs[0]))
```

运行命令：

```powershell
$env:PATH = 'C:\Program Files (x86)\sox-14-4-2;' + $env:PATH
f:\Lab\Qwen3-TTS\.venv\Scripts\python.exe .\manual_test.py
```

如果这个文件正常，但 Gradio 页面里还是噪音，那问题更可能在：

- Gradio 播放端
- 浏览器缓存
- 当前页面旧进程未重启

## 我建议你实际怎么跑

1. 先关掉现有 demo 进程。
2. 用 `CustomVoice + float16 + --no-flash-attn` 重新启动。
3. 先跑中文 `Vivian` 的空指令样本。
4. 再只改 `Instruction`，连续跑 3 到 4 条情绪对照。
5. 如果 UI 里还是噪音，就立刻做上面的 `soundfile` 脚本对照测试。

如果你手动跑完后，把以下信息发给我，我可以继续帮你定位：

- 你实际使用的启动命令
- 页面右侧 `Status` 文本框里的内容
- 哪个模型页正常，哪个模型页是噪音
- `custom_voice_manual_test.wav` 是否正常