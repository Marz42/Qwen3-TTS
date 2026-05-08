# MVP 实施方案

## 目标

基于 [mvp-design.md](mvp-design.md) 中定义的四层架构，落地一个可演示、可验收、可逐步扩展的本地 MVP：

1. GUI 层：Gradio
2. API 与服务层：FastAPI
3. 调度器与任务队列层：`subprocess` + 文件锁 / 进程锁
4. 模型与克隆音色库管理层：本地模型仓库 + SQLite + prompt 文件

这个 MVP 的重点不是一次性做全，而是建立一条完整闭环：

- 本地模型注册
- 模型加载与切换
- TTS 推理
- Base 模型音色提取与复用
- 训练数据准备
- 单说话人微调任务调度
- 训练结果回流到模型仓库

每一步都必须带有明确的测试方法和验收标准。

---

## 约束与前提

以下约束来自当前 fork 已验证行为，以及 [progress.md](progress.md)、[tech-stack.md](tech-stack.md)、[model-and-finetuning-guide.md](model-and-finetuning-guide.md) 中已经确认的事实：

- 当前主线能力应围绕 12Hz tokenizer 和 `Qwen3TTSModel` 展开。
- 本地推理已验证的稳妥配置是 Windows + CUDA + `float16` + `--no-flash-attn`。
- 仓库对 `transformers==4.57.3` 耦合较深，不应在 MVP 阶段改依赖版本。
- `CustomVoice`、`VoiceDesign`、`Base` 三类模型调用方式不同，服务层必须按 `tts_model_type` 分流。
- `0.6B-CustomVoice` 的 `instruct` 在当前代码里会被忽略，不能把它当成完整 instruct 模型。
- Base 音色克隆与 finetuning 产物不是同一种机制：前者保存 prompt，后者保存 checkpoint。
- `finetuning/` 当前只支持单说话人训练。
- 当前本机上 `1.7B` 推理已验证，但 Base 端到端和本机训练流程尚未完全验证，因此 MVP 应先以“功能通路 + 清晰错误处理”为目标，而不是先承诺完整生产性能。

---

## 风险评估与处理结论

下面这些风险需要视为 MVP 的一部分，而不是“后续优化”：

### 1. 推理并发导致 CUDA OOM

评估：高风险，而且是当前方案里最容易被低估的一项。

原因：即使没有训练任务，只要两个并发推理请求同时进入同一个 GPU 上的 `generate(...)`，也可能在显存很紧的机器上直接 OOM。当前本机只有 6 GB 显存，更不应该默认允许并发推理。

建议：

- MVP 阶段明确采用“单 GPU 单飞行”策略。
- 训练和推理不能并发。
- 推理和推理也不能并发。
- 第二个推理请求应被显式排队或直接返回 `429/503`，而不是抱着侥幸心理同时执行。

### 2. 子进程继续存活，但服务锁状态丢失

评估：高风险，且属于服务重启后最危险的状态不一致问题。

原因：如果训练脚本由子进程执行，而 FastAPI 主进程重启，内存中的 `gpu_lock` 会丢失，但训练进程可能还在继续占 GPU。

建议：

- 不要把 GPU 锁只做成内存变量。
- 使用落盘锁文件，例如 `data/jobs/gpu_training.lock`。
- 锁文件至少记录：`job_id`、`pid`、`created_at`、`type`。
- 每次加载模型、发起推理、启动训练前，都要检查锁文件与 PID 是否仍然有效。
- 当前是 Windows 环境，进程存活检查优先用 `psutil.pid_exists(pid)`；如果不引入 `psutil`，则至少提供一个平台兼容的 PID 校验实现。
- 服务启动时要执行一次“锁恢复检查”：清理脏锁，或识别仍在运行的训练任务并将系统置为 busy 状态。

### 3. 模型切换或请求处理时的显存泄漏

评估：高风险，但更像工程纪律问题，不能只靠 `torch.cuda.empty_cache()` 自信地认为已经解决。

建议：

- 服务层不要缓存推理输出 Tensor。
- 所有输出在离开服务层前都转换为 CPU 端 `numpy` 或落盘文件。
- 模型切换前显式 `del` 中间对象，并避免在闭包、全局变量、日志对象里意外保留 Tensor 引用。
- 将“多轮切换后显存是否持续上涨”纳入验收测试，而不是只测一次切换成功。

### 4. “至少 5 条样本”仍可能在训练阶段踩边界坑

评估：中高风险。

原因：`5` 只是业务下限，不代表训练参数一定安全。即使当前 `sft_12hz.py` 没有验证集划分逻辑，过小数据集配合不合适的 `batch_size`、未来的验证集扩展或步数统计，也可能导致训练质量差或任务脚本行为异常。

建议：

- 在训练启动前做 preflight 检查。
- 自动将 `batch_size` 截断为 `min(requested_batch_size, len(dataset))`。
- 显式计算 `effective_steps_per_epoch`，如果结果为 0 或明显异常，直接拒绝训练。
- MVP 阶段不引入验证集切分，先把训练链跑稳。

### 5. Base prompt 保存时的设备依赖问题

评估：高风险，而且一旦踩中会让 prompt 文件失去可移植性。

原因：`create_voice_clone_prompt(...)` 产生的张量很可能挂在 GPU 上。如果直接 `torch.save(...)`，后续读取容易和原始设备绑定，导致跨重启、跨设备或无 GPU 环境加载失败。

建议：

- 保存前将 prompt 内所有 Tensor 显式转到 CPU。
- 读取时使用 `map_location="cpu"`。
- 真正调用生成前，再由服务层按当前模型设备完成必要迁移。

### 6. 静态输出目录无限膨胀

评估：中风险，但几乎一定会发生。

建议：

- 在保存新 WAV 前做一次轻量清理。
- 以文件数量或目录总大小作为阈值。
- MVP 可以采用最简单策略：超过阈值时，按修改时间删除最旧的一批文件。

### 7. 子进程日志不可追踪

评估：高风险。

原因：一旦训练失败，若只有控制台输出，几乎无法在 GUI 或 API 中定位原因。

建议：

- 每个 job 单独建立目录，例如 `data/jobs/<job_id>/`。
- 训练子进程的 `stdout` 和 `stderr` 全部重定向到 `train.log`。
- `GET /api/v1/jobs/{job_id}` 应返回结构化状态和日志尾部摘要。

---

## MVP 范围冻结

### 本期必须做

- 模型仓库管理：登记本地 Base / CustomVoice / VoiceDesign 模型。
- 单实例模型管理器：任意时刻内存只保留一个模型。
- 通用 TTS 生成接口：按模型类型分发到三种生成函数。
- Base 音色提取、保存、复用。
- 训练任务触发、状态跟踪、训练完成后自动注册新模型。
- Gradio 作为纯 GUI，通过 HTTP 调 FastAPI。

### 本期明确不做

- 浏览器流式音频分块播放。
- 多说话人微调。
- 分布式任务队列、Celery、Redis。
- 在线下载模型作为主路径。
- 完整权限系统、多用户系统、对象存储。

---

## 推荐目录骨架

建议新增独立应用目录，例如 `app/`，避免把业务入口继续堆进 `qwen_tts/cli/demo.py`。

```text
app/
  api/
    main.py
    routes/
      models.py
      voices.py
      tts.py
      jobs.py
  services/
    model_manager.py
    tts_service.py
    voice_prompt_service.py
    training_service.py
    storage_service.py
  scheduler/
    job_runner.py
    lock_manager.py
  db/
    models.py
    schema.py
    session.py
  gui/
    gradio_app.py
  static/
    outputs/
  data/
    app_data.db
    prompts/
    datasets/
    jobs/
    pretrained_models/
```

这个目录结构不是唯一答案，但四层边界应尽量稳定：

- GUI 只负责界面和 HTTP 调用。
- API 只负责参数校验、状态码和服务编排。
- Scheduler 只负责长任务、锁和状态流转。
- 模型/音色库层只负责模型路径、prompt 文件、SQLite 元数据。

---

## 分阶段实施

## 阶段 0：冻结运行基线与本地资源布局

### 状态

- 已完成（2026-05-08）。
- 已在 `qwen_tts/app/runtime.py` 落地默认目录布局、默认推理参数、最小训练样本数和单 GPU 单飞行策略。
- `qwen_tts/cli/demo.py` 的默认并发已从原先的高并发假设收敛到 `concurrency=1`，并会显式拒绝大于 1 的值。

### 目标

先把 MVP 所依赖的本地资源和默认策略固定下来，避免后面一边开发一边变更运行假设。

### 实现项

1. 约定本地模型目录，例如 `data/pretrained_models/`。
2. 预下载至少 2 个本地模型：
   - 一个 Base 模型，优先 `0.6B-Base` 作为轻量闭环验证。
   - 一个非 Base 模型，优先 `1.7B-CustomVoice` 或 `1.7B-VoiceDesign`，因为当前已经验证过可运行。
3. 固定本地推理默认参数：
   - `device_map="cuda:0"`
   - `dtype=torch.float16`
   - `attn_implementation=None`
4. 明确静态文件目录：
   - `data/prompts/`
   - `data/jobs/`
   - `static/outputs/`
5. 明确训练数据最低门槛：一次训练至少 5 条音频样本，否则前端和 API 都拒绝启动训练。
6. 明确并发策略：MVP 阶段整个 GPU 只允许一个活动任务，无论它是训练还是推理。
7. 明确锁策略：训练锁必须落盘，不能只依赖内存中的 `gpu_lock`。

### 测试

1. 用绝对路径加载本地模型，而不是 repo id，验证离线可用。
2. 人工检查目录结构是否创建成功。
3. 用一个最小脚本从本地路径加载一次 `CustomVoice` 模型并完成单句推理。
4. 人工检查配置中已经明确写出“单 GPU 单飞行”策略，而不是默认允许并发推理。

### 验收标准

- 本地目录结构固定下来并写入配置。
- 至少 1 个本地路径模型完成成功加载和生成。
- 服务实现中不再把“动态下载模型”作为默认依赖路径。
- 并发与锁的基础策略已经在配置或实现骨架中固定，而不是留到后面临场处理。

### 本轮实现说明

- 已固定并实际创建 `data/pretrained_models/`、`data/prompts/`、`data/jobs/`、`static/outputs/`。
- 已验证运行基线模块可正常导入、可生成锁记录，并能输出完整基线配置。

---

## 阶段 1：落地模型与音色元数据层

### 状态

- 已完成（2026-05-08）。
- 已在 `qwen_tts/app/metadata.py` 落地 SQLite 元数据服务，并由 `build_metadata_store()` 自动初始化 `data/app_data.db`。

### 对应架构层

- 第 4 层：模型和克隆音色库管理

### 目标

先把 SQLite 与本地文件系统打通，让系统知道“有哪些模型”和“有哪些可复用音色”。

### 实现项

1. 创建 SQLite 数据库 `app_data.db`。
2. 创建 `models` 表，字段至少包括：
   - `id`
   - `name`
   - `type`
   - `path`
   - `speaker`
3. 创建 `voice_prompts` 表，字段至少包括：
   - `id`
   - `name`
   - `ref_text`
   - `prompt_file`
4. 写一个初始化脚本或启动时初始化逻辑：自动建表。
5. 写模型注册服务：支持手动录入本地模型路径。
6. 写 prompt 元数据服务：支持保存和查询 prompt 文件记录。

### 测试

1. 新建数据库后，查询两张表结构是否正确。
2. 手动插入一条 Base 模型记录和一条 CustomVoice 模型记录，再读回验证。
3. 手动插入一条 prompt 记录，再读回验证文件路径和 `ref_text`。
4. 重启服务后确认数据库内容仍可读。

### 验收标准

- 数据库能自动初始化。
- 至少 2 条模型记录可成功写入和查询。
- 至少 1 条音色 prompt 记录可成功写入和查询。
- 所有记录都以本地路径为准，不依赖网络拉取。

### 本轮实现说明

- `models` 表已按 `id/name/type/path/speaker` 建立。
- `voice_prompts` 表已按 `id/name/ref_text/prompt_file` 建立。
- 已通过一次本地往返验证完成两条模型记录和一条 prompt 记录的写入、读回和按 ID 查询。
- 注册逻辑当前要求路径为本地绝对路径，且模型目录与 prompt 文件必须实际存在。

---

## 阶段 2：落地单例 Model Manager

### 状态

- 已完成（2026-05-08）。
- 已在 `qwen_tts/app/model_manager.py` 落地单例 `ModelManager`，并通过假加载器完成复用、切换、GPU busy 和推理互斥的窄验证。

### 对应架构层

- 第 4 层：模型管理
- 第 2 层：服务层内部依赖

### 目标

建立“1 号换卡槽”策略：任意时刻显存中只保留一个模型，且训练、推理都共享同一套 GPU 访问控制。

### 实现项

1. 实现单例 `ModelManager`。
2. 维护以下状态：
   - `current_model_path`
   - `model`
   - `gpu_lock`
   - `inference_lock`
3. 实现 `load_model(path)`：
   - 如果当前模型就是目标路径，则直接复用。
   - 如果不同，则卸载旧模型、清缓存、加载新模型。
4. 把加载参数做成统一配置，不允许各路由各自随便写死。
5. 在服务层显式保留当前仓库已经修复过的安全路径：半精度主模型下 speech tokenizer 仍由底层保持 `float32`。
6. 推理调用必须放在显式互斥区内，MVP 阶段不允许两个推理请求同时调用 `generate(...)`。
7. 服务层输出必须在离开请求上下文前转换为 CPU 侧对象或文件，避免跨请求残留 GPU Tensor 引用。

### 测试

1. 连续两次加载同一个模型路径，验证返回的是复用路径而不是重复加载。
2. 先加载 `CustomVoice`，再切换到 `VoiceDesign`，验证旧模型已卸载且新模型可生成。
3. 手工记录切换前后的显存占用，确认没有明显累计泄漏。
4. 在 `gpu_lock=True` 时调用 `load_model(...)`，确认会返回受控错误而不是继续抢 GPU。
5. 并发发起两次推理请求，确认只有一个请求进入生成区，另一个请求被阻塞或被明确拒绝。
6. 连续多轮“加载 A -> 推理 -> 切换 B -> 推理 -> 切回 A”后，显存占用没有持续单调上涨。

### 验收标准

- 模型复用、卸载、切换三种路径都跑通。
- 模型切换后仍能完成一次实际推理。
- 训练锁开启时，推理请求不会偷偷加载模型。
- 并发推理不会直接并行打进 GPU 生成逻辑。
- 多轮切换后没有明显的显存泄漏趋势。

### 本轮实现说明

- `ModelManager` 已集中使用 `RuntimeBaseline` 的统一加载参数，不再允许调用方各自拼装 `device_map` / `dtype` / `attn_implementation`。
- 当前管理器已支持：
   - 同路径模型复用
   - 切换路径时卸载旧模型并清理缓存
   - 内存 `gpu_lock` 与磁盘 `gpu.lock` 的统一忙状态检查
   - 非阻塞推理互斥控制
   - 递归将 `torch.Tensor` 推理输出转成 CPU 侧对象
- 本轮验证使用假加载器完成，因此已经证明控制流正确，但尚未把真实 Base / CustomVoice / VoiceDesign 模型切换纳入同一轮自动验证。

---

## 阶段 3：落地 FastAPI 基础骨架与健康检查

### 状态

- 已完成（2026-05-08）。
- 已在 `qwen_tts/app/api/` 落地 FastAPI 应用工厂、健康检查、模型列表、音色列表、统一异常处理和静态文件挂载。

### 对应架构层

- 第 2 层：API 与服务层

### 目标

先把 API 壳子搭起来，提供稳定的健康检查、模型列表和错误返回格式。

### 实现项

1. 新建 FastAPI 应用入口。
2. 增加基础路由：
   - `GET /health`
   - `GET /api/v1/models/list`
   - `GET /api/v1/voices/list`
3. 统一异常处理：
   - 资源冲突返回 `409` 或 `503`
   - 参数错误返回 `400`
   - 未找到返回 `404`
4. 暴露静态文件目录，例如 `static/outputs/`。
5. 定义请求/响应 schema，避免把内部对象直接透传到外部。

### 测试

1. 启动 FastAPI 后访问 `/health`，确认服务存活。
2. 调 `/api/v1/models/list`，确认能返回数据库中的模型记录。
3. 调 `/api/v1/voices/list`，确认能返回空列表或已有列表。
4. 构造错误请求，验证状态码和错误消息是否一致。

### 验收标准

- API 可以稳定启动。
- 数据库列表接口可用。
- 错误返回格式固定，不再依赖 Python traceback 直出给前端。

### 本轮实现说明

- 已新增 `qwen_tts/app/api/main.py`、`schemas.py` 和 `routes/`。
- 当前已提供：
   - `GET /health`
   - `GET /api/v1/models/list`
   - `GET /api/v1/voices/list`
- 已挂载 `/static` 到仓库 `static/` 目录。
- 已通过 `TestClient` 验证：
   - `/health` 返回 `200`
   - 模型列表和音色列表接口返回结构化 JSON
   - `ValueError` 会被统一转成 `400`
   - `/static/outputs/phase2_base_validation.wav` 可被直接访问并返回 `audio/wav`

---

## 阶段 4：落地通用 TTS 生成服务

### 状态

- 已完成（2026-05-08）。
- 已完成真实模型三路 HTTP 验收，并补齐真实模型切换下的显存观测。

### 对应架构层

- 第 2 层：API 与服务层
- 第 4 层：模型管理层

### 目标

实现 `POST /api/v1/tts/generate`，按模型类型统一分发到三类推理函数。

### 实现项

1. 定义统一 payload，至少包含：
   - `model_id`
   - `text`
   - `language`
   - 可选 `speaker`
   - 可选 `instruct`
   - 可选 `prompt_id`
   - 可选 `ref_audio` / `ref_text`
2. 根据 `models.type` 分流：
   - `custom_voice` -> `generate_custom_voice(...)`
   - `voice_design` -> `generate_voice_design(...)`
   - `base` -> `generate_voice_clone(...)`
3. 输出统一处理：
   - 保存为 `static/outputs/<job-or-request-id>.wav`
   - 返回可访问 URL，而不是在 MVP 阶段做流式输出
4. 对参数做模型类型级别校验：
   - `custom_voice` 缺 `speaker` 直接报错
   - `voice_design` 缺 `instruct` 直接报错
   - `base` 缺 `ref_audio` 且缺 `prompt_id` 直接报错
   - `base` 且 `x_vector_only_mode=False` 但缺 `ref_text` 直接报错
5. 在生成新文件前，对 `static/outputs/` 做轻量回收：当文件数或总大小超过阈值时，按修改时间删除最旧的一批文件。
6. 推理接口若因并发策略无法立即执行，应返回受控状态码和明确消息，而不是等待到 OOM。

### 测试

1. 用一个 `CustomVoice` 模型完成一次生成，请求返回可播放静态 WAV URL。
2. 用一个 `VoiceDesign` 模型完成一次生成。
3. 用一个 `Base` 模型直接传 `ref_audio` + `ref_text` 完成一次生成。
4. 对三类模型分别提交错误 payload，验证返回的是受控错误。
5. 对同一个模型连续发两次请求，确认第二次会复用当前已加载模型。
6. 人工制造输出目录超过阈值，确认系统会删除最旧文件而不是无限堆积。
7. 并发提交两次推理，确认第二个请求收到受控拒绝或明确排队反馈。

### 验收标准

- 三类模型各至少 1 次成功生成。
- 输出 WAV 文件确实落盘并可通过 HTTP 访问。
- 错误参数不会导致服务崩溃。
- 输出目录存在受控清理机制。
- 并发推理不会因为竞争直接触发 OOM。

### 本轮实现说明

- 已新增 `qwen_tts/app/api/routes/tts.py` 和 `qwen_tts/app/tts_service.py`。
- 当前 `POST /api/v1/tts/generate` 已支持：
   - 统一 payload（`model_id`、`text`、`language`、`speaker`、`instruct`、`prompt_id`、`ref_audio`、`ref_text`）
   - 按 `models.type` 分流到 `generate_custom_voice` / `generate_voice_design` / `generate_voice_clone`
   - 模型类型级别参数校验
   - 输出 WAV 落盘并返回 `/static/outputs/...` URL
   - 生成前按阈值回收最旧输出文件
- 已通过假模型 `TestClient` 验证：
   - 三类分流请求可成功返回 `200`
   - 错误 payload（`custom_voice` 缺 `speaker`）返回 `400`
   - 静态输出 URL 可直接访问并返回 `audio/wav`
   - 在输出上限阈值为 3 时，连续 4 次生成后目录文件数仍保持 3
- 已完成真实模型 HTTP 验收（`CustomVoice` / `VoiceDesign` / `Base`）：
   - 四次请求 `custom -> voice_design -> base -> custom` 均返回 `200`
   - 四个输出 URL 均可访问并返回 `audio/wav`
   - 真实模型路径使用本地 Hugging Face snapshot 绝对路径
- 已完成真实模型切换显存观测：
   - `before_custom_http`: allocated/reserved = `0 / 0 MB`
   - `after_custom_http`: allocated/reserved = `4316.84 / 4890 MB`
   - `after_voice_design_http`: allocated/reserved = `4316.84 / 4324 MB`
   - `after_base_http`: allocated/reserved = `4404.49 / 4424 MB`
   - `after_custom_again_http`: allocated/reserved = `4316.84 / 4324 MB`
   - 结论：切换后显存回到同量级区间，没有出现线性累积上涨迹象。

---

## 阶段 5：落地 Base 音色提取与音色库复用

### 状态

- 已完成（2026-05-08）。
- 已通过真实 Base 模型完成全路径验收，含模拟重启后 `prompt_id` 复用与受控错误验证。

### 对应架构层

- 第 4 层：模型和克隆音色库管理
- 第 2 层：服务层

### 目标

支持从 Base 模型提取可复用音色 prompt，并将其保存到本地文件系统和 SQLite 中。

### 实现项

1. 新增 `POST /api/v1/voices/extract_prompt`。
2. 路由流程：
   - 接收 `model_id`
   - 校验该模型必须是 `base`
   - 接收 `ref_audio`、`ref_text`、`x_vector_only_mode`
   - 调用 `create_voice_clone_prompt(...)`
   - 保存前将 prompt 内所有 Tensor 迁移到 CPU
   - 用 `torch.save(...)` 保存为 `.pt`
   - 把文件路径和元数据写入 `voice_prompts` 表
3. 在 `POST /api/v1/tts/generate` 中支持 `prompt_id` 复用。
4. 读取 prompt 文件时统一使用 `map_location="cpu"`。
5. 为 prompt 文件读取失败、模型类型不匹配、设备不兼容等情况补充受控错误。

### 测试

1. 用 Base 模型提取一条 prompt 并保存。
2. 重启服务后通过 `prompt_id` 读取并生成一次语音。
3. 删除 prompt 文件后再调用，验证 API 返回受控错误而不是崩溃。
4. 用非 Base 模型调用 `extract_prompt`，验证被拒绝。
5. 在没有原始保存 GPU 上下文的情况下重新加载 prompt，确认仍可成功复用。

### 验收标准

- prompt `.pt` 文件真实落盘。
- 数据库中存在对应记录。
- 同一 prompt 可在服务重启后继续复用。
- 模型类型校验和文件缺失校验都有效。
- prompt 文件不与保存时的 GPU 设备强耦合。

### 本轮实现说明

- 已新增 `POST /api/v1/voices/extract_prompt`：
   - 新增请求/响应 schema：`VoicePromptExtractRequest`、`VoicePromptExtractResponse`
   - 接口校验 `model_id` 对应模型必须是 `base`
   - 校验 `ref_audio` 必填；`x_vector_only_mode=False` 时 `ref_text` 必填
- 已新增服务层 `qwen_tts/app/voice_service.py`：
   - 通过 `model.create_voice_clone_prompt(...)` 提取 prompt
   - 保存前递归将 prompt 结构内 Tensor 迁移到 CPU
   - 使用 `torch.save(...)` 落盘到 `data/prompts/*.pt`
   - 将 prompt 元数据写入 `voice_prompts` 表
- 已强化 `POST /api/v1/tts/generate` 的 prompt 读取错误处理：
   - 保持 `map_location="cpu"` 读取
   - prompt 文件损坏或加载异常时返回受控 `400`
- 已通过假模型 `TestClient` 窄验证（初轮）：
   - `extract_prompt`（Base 模型）返回 `200`
   - 非 Base 模型调用 `extract_prompt` 返回 `400`
   - 通过 `prompt_id` 调用 `tts/generate` 返回 `200`
   - 人工破坏 `.pt` 文件后再次生成返回 `400`（受控错误）
- 已通过真实 Base 模型完成 HTTP 验收（含模拟重启）：
   - `extract_prompt` 返回 `200`，`.pt` 文件落盘（19405 bytes）
   - 通过 `prompt_id` 生成返回 `200`，静态 URL 可访问 `audio/wav`
   - 模拟重启后（重新初始化 `MetadataStore` + `ModelManager`）同一 `prompt_id` 仍返回 `200`
   - 损坏 `.pt` 文件后生成返回 `400`（受控错误）
- 修复了 PyTorch >= 2.4 `weights_only=True` 默认行为导致的 `VoiceClonePromptItem` 无法 pickle 加载问题：
   - 保存时将 `List[VoiceClonePromptItem]` 序列化为纯 dict（仅含 Tensor + Python 基本类型）
   - 加载时用 `weights_only=True` 明确读取，随后重建 `VoiceClonePromptItem` 对象

---

## 阶段 6：落地调度器与训练任务状态机

### 对应架构层

## 阶段 6：落地调度器与训练任务状态机
- 第 2 层：服务层
### 状态
- 第 4 层：模型管理层
- 已完成 TestClient 窄验证（2026-05-09）。
- 8 项验证全部通过：状态机流转、磁盘锁、启动恢复、并发拒绝、503 推理拦截、log_tail 返回。
- 真实 `sft_12hz.py` 训练受 `flash_attention_2` 限制，当前验收使用 fake subprocess runner；调度控制流已完整验证。

### 对应架构层
### 目标
- 第 3 层：调度器与任务队列

让长时间训练不阻塞 API，并且保证训练与推理不会同时争抢同一块 GPU。

### 实现项

1. 定义任务状态：
   - `pending`
   - `running`
   - `succeeded`
   - `failed`
2. 新增训练任务记录存储，最简单可以先用 `jobs/` 目录里的 JSON 文件或新增 SQLite 表。
3. 新增 `POST /api/v1/models/train`：
   - 校验样本数不少于 5 条
   - 做训练前 preflight：校验 `batch_size <= len(dataset)`，必要时自动下调；计算 `effective_steps_per_epoch`
   - 校验训练输入目录和 JSONL 是否存在
   - 抢占落盘训练锁，而不是只改内存变量
   - 卸载当前内存模型
   - 后台线程启动 `prepare_data.py` 和 `sft_12hz.py`
4. 新增 `GET /api/v1/jobs/{job_id}` 用于查看任务状态和日志摘要。
5. 训练完成后自动把新 checkpoint 写入 `models` 表。
6. 训练期间所有推理接口直接返回 `503`，明确提示“系统正在训练中”。
7. 失败时确保：
   - 锁被释放
   - 任务状态进入 `failed`
   - 日志可追踪
8. 每个 job 都有独立目录，例如 `data/jobs/<job_id>/`，并至少包含：
   - `metadata.json`
   - `train.log`
   - 可选的 `prepare.log`
9. 子进程标准输出和错误都重定向到日志文件。
10. 服务启动时执行锁恢复检查：
   - 锁文件不存在则视为空闲
   - 锁文件存在且 PID 仍活跃，则视为 busy
   - 锁文件存在但 PID 已失效，则清理脏锁并记录恢复日志

### 测试

1. 构造一个最小训练任务，验证任务能进入 `running`。
2. 在训练运行时调用 TTS 接口，确认返回 `503`。
3. 人为制造一个错误参数训练任务，确认任务进入 `failed`，且锁被释放。
4. 训练完成后检查数据库，确认新增了一条 `custom_voice` 模型记录。
5. 用新增模型记录发起一次推理，确认 checkpoint 可被真实加载。
6. 在训练运行中重启 FastAPI，确认服务能根据锁文件与 PID 状态恢复 busy 状态，而不是误判为空闲。
7. 人为制造脏锁文件，确认启动恢复逻辑可以识别并清理。
8. 查看 `GET /api/v1/jobs/{job_id}`，确认至少能返回日志尾部最后 50 行。
9. 构造 5 条样本的小数据集，验证 preflight 之后训练参数仍然有效，任务不会因为批大小配置明显不合理而直接失控。

### 验收标准

- 训练任务不会阻塞 API 主线程。
- 训练期间推理被正确拒绝。
- 成功与失败两种任务路径都能正确释放锁。
- 成功训练后，新模型能自动进入模型仓库并可调用。
- 服务重启后不会因为内存锁丢失而误判 GPU 空闲。
- 训练日志可以通过 API 追踪，而不是只存在终端滚动输出中。
- 小样本训练在 preflight 后能够得到明确的可执行参数或明确的拒绝理由。

---

## 阶段 7：落地数据准备 GUI

### 对应架构层

- 第 1 层：GUI 层
- 第 2 层：API 层

### 目标

为训练准备提供一个最小但完整的可视化入口，确保 GUI 只负责交互，不承载业务核心逻辑。

### 实现项

1. 新建 Gradio 页面 A：数据准备与微调。
2. 页面支持：
   - 上传音频文件或压缩包
   - 展示切分结果
   - 展示 ASR 结果
   - 人工修正文案
   - 生成 `train_raw.jsonl`
   - 触发训练任务
3. GUI 中所有按钮都通过 HTTP 调 FastAPI，不直接操作数据库和模型对象。
4. 明确界面约束：少于 5 条音频不允许提交训练。
5. 如果暂时未接入 SenseVoice-Small，MVP 第一版允许先用手动文本录入替代，但接口层要预留 ASR 步骤位置。

### 测试

1. 从 GUI 上传一组最小样本，确认前端能看到待编辑条目。
2. 修正文案后提交，确认后端生成 `train_raw.jsonl`。
3. 少于 5 条样本时提交训练，确认前端显示明确错误。
4. 点击训练按钮后，确认 GUI 是通过 HTTP 收到 job id，而不是页面线程卡死。

### 验收标准

- GUI 能完成一轮最小数据准备和训练提交。
- 前端不直接耦合数据库和模型对象。
- 训练前的数据量约束在 GUI 和 API 两端都生效。

---

## 阶段 8：落地 TTS 推理 GUI

### 对应架构层

- 第 1 层：GUI 层
- 第 2 层：API 层

### 目标

把模型选择、普通 TTS、Base prompt 复用和结果试听全部接到同一套 HTTP 服务上。

### 实现项

1. 新建 Gradio 页面 B：TTS 推理测试。
2. 支持：
   - 列出模型仓库中的模型
   - 根据模型类型动态显示必要参数
   - 选择 `prompt_id` 进行 Base 克隆
   - 播放或下载生成结果
3. 对训练中状态做前端提示：如果 API 返回 `503`，页面明确显示“资源被训练任务占用”。

### 测试

1. 页面加载时能正确拉取模型列表。
2. 选择 `custom_voice` 模型时出现 `speaker` 字段。
3. 选择 `voice_design` 模型时出现 `instruct` 字段。
4. 选择 `base` 模型时出现 `prompt_id` 或参考音频输入区域。
5. 成功生成后能在线播放或下载 WAV。

### 验收标准

- GUI 参数项能随模型类型变化。
- 推理页面完整走 HTTP API，不直接触碰底层模型对象。
- 用户可以在单个页面里完成模型选择、参数填写、试听结果。

---

## 阶段 9：端到端验收与回归清单

### 目标

在 MVP 交付前，至少完成一次跨四层的完整闭环验收。

### 测试

### 端到端验收路径 A：推理闭环

1. 在数据库中已有至少 1 个 `custom_voice` 模型和 1 个 `base` 模型。
2. 从推理 GUI 选择 `custom_voice` 模型，生成 1 条音频。
3. 从推理 GUI 选择 `base` 模型，上传参考音频并生成 1 条音频。
4. 从 Base 中提取并保存 1 条 prompt。
5. 使用该 prompt 再次生成 1 条音频。

### 端到端验收路径 B：训练闭环

1. 从数据准备 GUI 上传并整理至少 5 条样本。
2. 生成 `train_raw.jsonl`。
3. 触发训练任务。
4. 轮询 job 状态直到完成。
5. 检查新模型是否自动写入模型仓库。
6. 用新模型完成 1 次 `generate_custom_voice(...)` 推理。

### 回归检查项

1. 训练结束后训练锁已释放，且不存在脏锁文件。
2. 再次调用旧模型或新模型推理时， Model Manager 能正常切换。
3. prompt 文件和输出音频文件路径都可追踪。
4. 数据库没有出现模型记录存在但文件缺失、或文件存在但数据库无记录的明显脏数据。
5. 并发推理压测下系统返回受控拒绝或串行化执行，而不是直接 OOM。
6. 重启服务后，训练任务状态、锁状态和日志状态仍可恢复。
7. 输出目录清理策略生效，没有出现静态目录无限膨胀。

### 验收标准

- 两条端到端路径至少各成功一遍。
- 失败路径也至少验证一遍：训练失败、prompt 文件缺失、参数不全。
- 风险回归路径也至少验证一遍：并发推理、服务重启后的锁恢复、输出目录清理、日志追踪。
- MVP 交付时应附带一份人工验收记录，明确哪条能力已经通过、哪条能力仍受硬件或脚本默认配置限制。

---

## 实施顺序建议

按下面顺序推进最稳妥：

1. 阶段 0：固定本地资源与约束
2. 阶段 1：数据库与文件仓库
3. 阶段 2：Model Manager
4. 阶段 3：FastAPI 基础骨架
5. 阶段 4：通用 TTS 生成
6. 阶段 5：Base 音色提取与复用
7. 阶段 6：训练调度
8. 阶段 7：数据准备 GUI
9. 阶段 8：推理 GUI
10. 阶段 9：端到端验收

原因是：

- 如果第 4 层和第 2 层没站稳，GUI 只是堆积复杂度。
- 如果推理闭环没先跑通，训练闭环的调试成本会非常高。
- 如果训练调度没有明确状态机和资源锁，GUI 很容易把问题藏起来而不是解决掉。

---

## 当前最现实的第一批验收目标

结合当前本地验证进度，第一批建议先争取完成这 4 项：

1. 本地数据库 + 模型仓库可用。
2. FastAPI + Model Manager 能跑通 `CustomVoice` 和 `VoiceDesign` 推理。
3. Base prompt 提取、保存、复用闭环跑通。
4. 训练调度至少跑通“启动任务、占锁、失败释放锁、成功后注册模型”这条链。

只有这四项都站稳了，后面的 Gradio 双页面才值得往上搭。