# Model and Fine-Tuning Guide (Fork Local)

## 1. 三类模型调用

### 1.1 CustomVoice

- 接口：`generate_custom_voice`
- 必填：`text`、`speaker`
- 可选：`language`、`instruct`

### 1.2 VoiceDesign

- 接口：`generate_voice_design`
- 必填：`text`、`instruct`
- 可选：`language`

### 1.3 Base

- 接口：`generate_voice_clone`
- 两种模式：
  1. `ref_audio` + `ref_text`
  2. `prompt_id`（先通过 `extract_prompt` 提取）

## 2. Prompt 提取与复用

- 提取接口：`POST /api/v1/voices/extract_prompt`
- 复用方式：`POST /api/v1/tts/generate` 带 `prompt_id`
- 文件落盘：`data/prompts/*.pt`
- 元数据：`voice_prompts` 表

### 2.1 兼容性说明

- 保存时将张量转 CPU。
- 读取时使用 `weights_only=True`，随后重建 `VoiceClonePromptItem`。

## 3. 微调流程（当前能力）

- 当前仅支持单说话人训练。
- 链路：
  1. `POST /api/v1/data/collect_samples`
  2. `POST /api/v1/data/build_train_jsonl`
  3. `POST /api/v1/models/train`
  4. `GET /api/v1/jobs/{job_id}`

训练后台由 `JobManager` 执行：

- `prepare_data.py`
- `sft_12hz.py`
- 训练成功后自动注册 `custom_voice` 模型记录

## 4. 参数约束

- 样本数必须 >= 5。
- 训练模型必须是 `base` 类型。
- `batch_size` 会自动截断为不超过样本数。

## 5. 环境限制

- 当前本机真实训练受 `flash_attention_2` 限制。
- 因此已通过的验收侧重控制流和 API 行为，不代表真实吞吐性能。

## 6. 建议的验收顺序

1. 先跑 Phase 6-9 脚本验证服务层闭环。
2. 再在支持环境上做真实训练吞吐与质量评估。
3. 最后评估 0.6B 路径在本机的速度与显存行为。
