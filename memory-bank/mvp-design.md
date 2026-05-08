
### 1. 系统总体架构选型

*   **核心后端服务：FastAPI**（负责提供标准的 RESTful API、维护模型管理器单例、提供全局锁）。
*   **前端与可视化：Gradio**（分为两个独立页面：数据准备与微调 GUI，以及 TTS 推理测试 GUI。Gradio 通过内部 HTTP 调用 FastAPI 的接口）。
*   **数据库：SQLite**（单文件数据库，零配置配置）。
*   **长任务处理：Python `subprocess` + 文件锁**（抛弃庞大的 Celery，直接开子进程跑脚本，用极简的互斥锁防止 OOM）。
*   **ASR 组件：SenseVoice-Small** (同属阿里生态，速度极快，适合做音频预标注)。


四层架构：

1. GUI层：Gradio
2. API与服务层：FastAPI
3. 调度器与任务队列
4. 模型和克隆音色库管理

---

### 2. 核心模块详细设计

#### 2.1 数据库设计 (SQLite: `app_data.db`)

建两张极简表，用来管理模型和音色：

**表 1：`models` (模型仓库)**
| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `id` | Integer (PK) | 唯一ID |
| `name` | String | 展示名称，如 "张三微调V1" |
| `type` | String | 模型类型：`base`, `voice_design`, `custom_voice` |
| `path` | String | 物理路径：`Qwen/Qwen3-TTS...` 或 `./output/ckpt-2` |
| `speaker` | String | 绑定的 `speaker_name`（CustomVoice 专属） |

**表 2：`voice_prompts` (克隆音色库)**
| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `id` | Integer (PK) | 唯一ID |
| `name` | String | 音色名称，如 "清冷女声" |
| `ref_text` | String | 参考文本（ICL 模式需要记录） |
| `prompt_file` | String | 本地 `.pt` 文件路径，存放 `create_voice_clone_prompt` 提取的张量 |

#### 2.2 模型管理器 (Model Manager) —— “1号换卡槽”策略

按照你的思路，我们实现一个**单例模式（Singleton）**的模型管理器。内存中永远只保留一个模型。

```python
import torch
import gc
from qwen_tts import Qwen3TTSModel

class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.current_model_path = None
            cls._instance.model = None
            cls._instance.gpu_lock = False # 全局 GPU 锁（防训练和推理撞车）
        return cls._instance

    def load_model(self, model_path: str):
        # 1. 检查是否正在训练
        if self.gpu_lock:
            raise Exception("GPU is currently locked by a training task.")

        # 2. 如果路径相同，直接复用
        if self.current_model_path == model_path and self.model is not None:
            return self.model

        # 3. 卸载旧模型，清空显存
        if self.model is not None:
            del self.model
            gc.collect()
            torch.cuda.empty_cache()

        # 4. 加载新模型
        print(f"Loading model: {model_path}")
        self.model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" 
        )
        self.current_model_path = model_path
        return self.model
```

#### 2.3 TTS 服务与音色持久化

**TTS 生成逻辑：**
FastAPI 接收到请求后，查询 SQLite 获取模型 `path`，调用 ModelManager 加载模型并生成音频。

**克隆音色提取与保存 (`Base` 模型)：**
1. 接收 `ref_audio` 和 `ref_text`。
2. 调用 `tts.create_voice_clone_prompt()`。
3. 关键点：返回的 prompt 实际上包含了 Tensor，直接用 `torch.save(prompt, f"./prompts/{prompt_id}.pt")` 存入本地文件系统。
4. 将 `{prompt_id}.pt` 的路径存入 SQLite 的 `voice_prompts` 表。
5. 下次推理时：`prompt = torch.load(f"./prompts/{prompt_id}.pt")`，直接传给 `generate_voice_clone(..., voice_clone_prompt=prompt)`。

#### 2.4 数据准备与微调调度 (Scheduler MVP 版)

**数据准备流程 (Gradio GUI)：**
1. **上传：** 用户在 Gradio 上传音频文件或压缩包。
2. **切分与转换：** 后台用 `pydub` 或 `librosa` 强制转成 24kHz 单声道 WAV，并按静音切分成短句。
3. **ASR 预打标：** 调用本地的 SenseVoice 快速识别生成文字。
4. **人工纠正：** 将切分好的 `[音频, 识别文本]` 对以 Gradio Dataframe 或列表形式展示，用户修改文本。
5. **保存：** 确认无误后，后台生成 `train_raw.jsonl`，并自动选定一条清晰音频作为 `ref.wav`。

**极简调度器 (Subprocess + Lock)：**
为了不让长达几小时的训练阻塞 FastAPI 接口响应，并防止显存爆炸：

```python
import subprocess
import threading

def start_training_task(train_jsonl, output_dir, speaker_name):
    manager = ModelManager()
    if manager.gpu_lock:
        return {"status": "error", "msg": "Another training is running"}
    
    # 1. 抢占全局锁并卸载当前内存模型
    manager.gpu_lock = True
    if manager.model is not None:
        del manager.model
        torch.cuda.empty_cache()
        manager.current_model_path = None

    # 2. 异步执行子进程
    def run_scripts():
        try:
            # 跑预处理
            subprocess.run(["python", "prepare_data.py", "--input...", ...], check=True)
            # 跑微调
            subprocess.run(["python", "sft_12hz.py", "--train...", ...], check=True)
            # 训练完毕，存入 SQLite models 表
            # db.execute("INSERT INTO models ...")
        finally:
            # 无论成功失败，释放锁
            manager.gpu_lock = False
            
    threading.Thread(target=run_scripts).start()
    return {"status": "started", "msg": "Training started in background"}
```
*(注意：MVP 阶段如果用户在训练期间请求 TTS 推理 API，API 直接返回 HTTP 503 "系统正在训练中，资源不可用"。)*

#### 2.5 API 路由设计规范

**`/api/v1/models`**
*   `GET /list` -> 读取 SQLite `models` 表。
*   `POST /train` -> 触发上述异步子进程。

**`/api/v1/voices`**
*   `POST /extract_prompt` -> 传入音频，提取并保存 `.pt`，记录入 SQLite。
*   `GET /list` -> 获取已存音色列表。

**`/api/v1/tts`**
*   `POST /generate` -> 通用接口。
    *   Payload 中带 `model_id`。
    *   根据模型类型，如果 type=`base` 且带了 `prompt_id`，就从本地加载 `.pt` 进行克隆。
    *   如果 type=`custom_voice`，就查询模型的 `speaker` 进行调用。

---

### 3. 给你的 MVP 落地建议（排坑指南）

1. **别用 HuggingFace/ModelScope 动态下载做 MVP：** 
   *   把 `0.6B-Base`、`0.6B-CustomVoice` 等基础模型全部提前下载到本地硬盘的 `./pretrained_models/` 目录下。
   *   数据库里存绝对路径或相对路径。不要让 API 在第一次调用时因为网络原因卡死在下载上。对于 MVP 来说，0.6B 版本已经足够验证闭环。
2. **Gradio 前后端分离的假象：**
   *   不要在 Gradio 的回调函数里直接写死业务逻辑（例如读写 DB）。
   *   Gradio 里的按钮应该通过 `requests.post("http://localhost:8000/api/v1/...")` 去调用你的 FastAPI。这样你的“API服务”不仅是给外部应用准备的，连你自己的 GUI 也在吃这套 API（Dogfooding）。
3. **不要纠结流式 (Streaming) 返回：**
   *   MVP 阶段，直接生成完整的 WAV 文件，存盘后向前端返回一个可以通过 HTTP 播放的静态 URL（例如 `/static/output/123.wav`）。流式音频分块（Chunking）在客户端组装会有大量浏览器兼容性坑，会极大拖慢 MVP 的交付速度。
4. **训练数据的“一键保底”：**
   *   GUI 必须有个约束：每次训练哪怕只有 5 条音频，也必须要有！