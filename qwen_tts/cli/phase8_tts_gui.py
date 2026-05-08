from __future__ import annotations

import argparse
import json
import tempfile
import uuid
from pathlib import Path
from typing import Any
from urllib import error, request

import gradio as gr


def _http_json(method: str, url: str, payload: dict | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = request.Request(url=url, method=method.upper(), headers=headers, data=data)
    try:
        with request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(body).get("detail", body)
        except Exception:
            detail = body
        raise ValueError(f"HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise ValueError(f"Cannot connect to API server: {exc.reason}") from exc


def _resolve_models(api_base: str) -> tuple[list[str], dict[str, dict[str, Any]]]:
    endpoint = api_base.rstrip("/") + "/api/v1/models/list"
    models = _http_json("GET", endpoint)
    mapping: dict[str, dict[str, Any]] = {}
    labels: list[str] = []
    for item in models:
        label = f"{item['id']} | {item['type']} | {item['name']}"
        labels.append(label)
        mapping[label] = item
    return labels, mapping


def _resolve_prompts(api_base: str) -> tuple[list[str], dict[str, int]]:
    endpoint = api_base.rstrip("/") + "/api/v1/voices/list"
    prompts = _http_json("GET", endpoint)
    labels = ["(不使用 prompt_id)"]
    mapping: dict[str, int] = {}
    for item in prompts:
        label = f"{item['id']} | {item['name']}"
        labels.append(label)
        mapping[label] = int(item["id"])
    return labels, mapping


def _refresh_resources(api_base: str):
    model_labels, model_map = _resolve_models(api_base)
    prompt_labels, prompt_map = _resolve_prompts(api_base)

    model_value = model_labels[0] if model_labels else None
    prompt_value = prompt_labels[0] if prompt_labels else None
    msg = f"已加载模型 {len(model_labels)} 个，prompt {len(prompt_labels) - 1} 个。"

    return (
        gr.update(choices=model_labels, value=model_value),
        model_map,
        gr.update(choices=prompt_labels, value=prompt_value),
        prompt_map,
        msg,
    )


def _model_type_updates(selected_label: str | None, model_map: dict[str, dict[str, Any]]):
    if not selected_label or selected_label not in model_map:
        hidden = gr.update(visible=False)
        return "未选择模型。", hidden, hidden, hidden, hidden, hidden

    model = model_map[selected_label]
    model_type = model.get("type", "")
    info = (
        f"model_id={model.get('id')}\n"
        f"name={model.get('name')}\n"
        f"type={model_type}\n"
        f"speaker(default)={model.get('speaker')}"
    )

    show_speaker = gr.update(visible=(model_type == "custom_voice"))
    show_instruct = gr.update(visible=(model_type == "voice_design"))
    show_prompt = gr.update(visible=(model_type == "base"))
    show_ref_audio = gr.update(visible=(model_type == "base"))
    show_ref_text = gr.update(visible=(model_type == "base"))

    return info, show_speaker, show_instruct, show_prompt, show_ref_audio, show_ref_text


def _download_audio_file(full_url: str) -> str:
    out_dir = Path(tempfile.gettempdir()) / "qwen3_tts_phase8"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{uuid.uuid4()}.wav"

    req = request.Request(full_url, method="GET")
    with request.urlopen(req, timeout=60) as resp:
        out_path.write_bytes(resp.read())
    return str(out_path)


def _generate_tts(
    api_base: str,
    selected_label: str,
    model_map: dict[str, dict[str, Any]],
    prompt_label: str,
    prompt_map: dict[str, int],
    text: str,
    language: str,
    speaker: str,
    instruct: str,
    ref_audio: str,
    ref_text: str,
    x_vector_only_mode: bool,
):
    if not selected_label or selected_label not in model_map:
        return None, "", "请先选择模型。"
    if not text.strip():
        return None, "", "text 不能为空。"

    model = model_map[selected_label]
    model_type = model.get("type")

    payload: dict[str, Any] = {
        "model_id": int(model["id"]),
        "text": text.strip(),
        "language": language.strip() if language.strip() else "Auto",
    }

    if model_type == "custom_voice":
        payload["speaker"] = speaker.strip()
    elif model_type == "voice_design":
        payload["instruct"] = instruct.strip()
    elif model_type == "base":
        payload["x_vector_only_mode"] = bool(x_vector_only_mode)
        prompt_id = prompt_map.get(prompt_label)
        if prompt_id is not None:
            payload["prompt_id"] = prompt_id
        else:
            payload["ref_audio"] = ref_audio.strip()
            payload["ref_text"] = ref_text.strip()
    else:
        return None, "", f"不支持的模型类型: {model_type}"

    endpoint = api_base.rstrip("/") + "/api/v1/tts/generate"
    try:
        result = _http_json("POST", endpoint, payload)
    except ValueError as exc:
        return None, "", str(exc)

    urls = result.get("output_urls") or []
    if not urls:
        return None, "", "API 未返回 output_urls。"

    first = str(urls[0])
    full_url = first if first.startswith("http") else api_base.rstrip("/") + first

    try:
        audio_path = _download_audio_file(full_url)
    except Exception as exc:
        return None, full_url, f"生成成功，但下载音频失败：{exc}"

    return audio_path, full_url, f"生成成功：request_id={result.get('request_id')}"


def create_phase8_tts_gui(default_api_base: str = "http://127.0.0.1:8010") -> gr.Blocks:
    with gr.Blocks(title="Qwen3-TTS Phase 8 TTS GUI") as demo:
        gr.Markdown(
            """
# Qwen3-TTS Phase 8 - 推理 GUI

本页面仅通过 HTTP 调 FastAPI：
1) 拉取模型列表与 prompt 列表
2) 按模型类型动态展示参数
3) 调用 `/api/v1/tts/generate` 并播放/下载结果

当 API 返回 503 时，页面会显示训练占用提示。
"""
        )

        api_base = gr.Textbox(label="FastAPI Base URL", value=default_api_base)
        refresh_btn = gr.Button("刷新模型与 prompt")

        model_choices_state = gr.State({})
        prompt_choices_state = gr.State({})

        model_select = gr.Dropdown(label="模型选择", choices=[], value=None)
        model_info = gr.Textbox(label="模型信息", lines=4)

        text = gr.Textbox(label="text", lines=4)
        language = gr.Textbox(label="language", value="Auto")

        speaker = gr.Textbox(label="speaker (custom_voice)", visible=False)
        instruct = gr.Textbox(label="instruct (voice_design)", lines=3, visible=False)

        prompt_select = gr.Dropdown(
            label="prompt_id (base，可选)",
            choices=["(不使用 prompt_id)"],
            value="(不使用 prompt_id)",
            visible=False,
        )
        ref_audio = gr.Textbox(label="ref_audio (base，未使用 prompt_id 时必填)", visible=False)
        ref_text = gr.Textbox(label="ref_text (base，x_vector_only_mode=False 时必填)", visible=False)
        x_vector_only_mode = gr.Checkbox(label="x_vector_only_mode (base)", value=False)

        generate_btn = gr.Button("生成语音")

        audio_out = gr.Audio(label="试听 / 下载", type="filepath")
        output_url = gr.Textbox(label="输出 URL")
        status = gr.Textbox(label="状态")

        refresh_btn.click(
            fn=_refresh_resources,
            inputs=[api_base],
            outputs=[model_select, model_choices_state, prompt_select, prompt_choices_state, status],
        )

        model_select.change(
            fn=_model_type_updates,
            inputs=[model_select, model_choices_state],
            outputs=[model_info, speaker, instruct, prompt_select, ref_audio, ref_text],
        )

        generate_btn.click(
            fn=_generate_tts,
            inputs=[
                api_base,
                model_select,
                model_choices_state,
                prompt_select,
                prompt_choices_state,
                text,
                language,
                speaker,
                instruct,
                ref_audio,
                ref_text,
                x_vector_only_mode,
            ],
            outputs=[audio_out, output_url, status],
        )

    return demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-tts-phase8-tts-gui",
        description="Launch Phase 8 TTS GUI that talks to FastAPI over HTTP.",
    )
    parser.add_argument("--api-base", default="http://127.0.0.1:8010", help="FastAPI base URL")
    parser.add_argument("--ip", default="127.0.0.1", help="Gradio bind IP")
    parser.add_argument("--port", type=int, default=8030, help="Gradio bind port")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    demo = create_phase8_tts_gui(default_api_base=args.api_base)
    demo.launch(server_name=args.ip, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
