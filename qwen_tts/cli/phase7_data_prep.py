from __future__ import annotations

import argparse
import json
from typing import Any, List
from urllib import error, request

import gradio as gr


def _http_json(method: str, url: str, payload: dict | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = request.Request(url=url, method=method.upper(), headers=headers, data=data)

    try:
        with request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
            detail = parsed.get("detail", body)
        except Exception:
            detail = body
        raise ValueError(f"HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise ValueError(f"Cannot connect to API server: {exc.reason}") from exc


def _build_rows(files: list[Any] | None) -> list[list[str]]:
    rows: list[list[str]] = []
    for f in files or []:
        path = getattr(f, "name", None) or str(f)
        rows.append([path, "", ""])
    return rows


def _collect_samples(
    api_base: str,
    audio_files: list[Any] | None,
    archive_files: list[Any] | None,
    use_asr_placeholder: bool,
) -> tuple[list[list[str]], str]:
    audio_paths: list[str] = []
    for f in audio_files or []:
        path = getattr(f, "name", None) or str(f)
        if path:
            audio_paths.append(path)

    archive_paths: list[str] = []
    for f in archive_files or []:
        path = getattr(f, "name", None) or str(f)
        if path:
            archive_paths.append(path)

    if not audio_paths and not archive_paths:
        return [], "请先上传音频或 zip 压缩包。"

    payload = {
        "audio_files": audio_paths,
        "archives": archive_paths,
        "use_asr_placeholder": bool(use_asr_placeholder),
    }
    endpoint = api_base.rstrip("/") + "/api/v1/data/collect_samples"
    data = _http_json("POST", endpoint, payload)

    samples = data.get("samples") or []
    rows: list[list[str]] = []
    for sample in samples:
        rows.append([
            str(sample.get("audio") or ""),
            str(sample.get("text") or ""),
            str(sample.get("asr_text") or ""),
        ])

    imported_dir = data.get("imported_dir")
    sample_count = int(data.get("sample_count") or len(rows))
    low_sample_tip = "样本数少于 5，训练提交时会被后端拒绝。" if sample_count < 5 else "可提交训练。"
    if imported_dir:
        msg = f"已收集 {sample_count} 条样本（含 zip 解包目录：{imported_dir}）。{low_sample_tip}"
    else:
        msg = f"已收集 {sample_count} 条样本。{low_sample_tip}"
    return rows, msg


def _save_train_jsonl(api_base: str, output_name: str, table_rows: list[list[str]] | None) -> tuple[str, str]:
    rows = table_rows or []
    samples = []
    for idx, row in enumerate(rows):
        if not row or len(row) < 2:
            continue
        audio = (row[0] or "").strip()
        text = (row[1] or "").strip()
        asr_text = (row[2] or "").strip() if len(row) >= 3 and row[2] else None
        if not audio:
            continue
        if not text:
            return "", f"第 {idx + 1} 行 text 为空，请先完善文案。"
        sample = {"audio": audio, "text": text}
        if asr_text:
            sample["asr_text"] = asr_text
        samples.append(sample)

    if not samples:
        return "", "没有可保存的样本，请先上传音频并填写文案。"

    payload = {"samples": samples}
    if output_name.strip():
        payload["output_name"] = output_name.strip()

    endpoint = api_base.rstrip("/") + "/api/v1/data/build_train_jsonl"
    data = _http_json("POST", endpoint, payload)
    output_jsonl = data.get("output_jsonl", "")
    sample_count = int(data.get("sample_count", 0))
    msg = f"已生成 train_raw.jsonl，共 {sample_count} 条：{output_jsonl}"
    return output_jsonl, msg


def _submit_training(
    api_base: str,
    base_model_id: int,
    speaker_name: str,
    input_jsonl: str,
    num_epochs: int,
    batch_size: int,
    lr: float,
) -> tuple[str, str]:
    if not input_jsonl.strip():
        return "", "请先生成 train_raw.jsonl。"
    if base_model_id <= 0:
        return "", "base_model_id 必须大于 0。"
    if not speaker_name.strip():
        return "", "speaker_name 不能为空。"

    payload = {
        "base_model_id": int(base_model_id),
        "speaker_name": speaker_name.strip(),
        "input_jsonl": input_jsonl.strip(),
        "num_epochs": int(num_epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
    }

    endpoint = api_base.rstrip("/") + "/api/v1/models/train"
    data = _http_json("POST", endpoint, payload)
    job_id = data.get("job_id", "")
    status = data.get("status", "")
    return job_id, f"训练任务已提交：job_id={job_id}, status={status}"


def _query_job(api_base: str, job_id: str) -> str:
    if not job_id.strip():
        return "请先提交训练任务并获得 job_id。"
    endpoint = api_base.rstrip("/") + f"/api/v1/jobs/{job_id.strip()}"
    data = _http_json("GET", endpoint)

    status = data.get("status")
    started_at = data.get("started_at")
    finished_at = data.get("finished_at")
    error_msg = data.get("error")
    log_tail = data.get("log_tail") or ""

    lines = [
        f"status: {status}",
        f"started_at: {started_at}",
        f"finished_at: {finished_at}",
    ]
    if error_msg:
        lines.append(f"error: {error_msg}")

    lines.append("\n--- log_tail (last 50 lines) ---")
    lines.append(log_tail if log_tail else "(empty)")
    return "\n".join(lines)


def create_phase7_data_prep_demo(default_api_base: str = "http://127.0.0.1:8010") -> gr.Blocks:
    with gr.Blocks(title="Qwen3-TTS Phase 7 Data Prep") as demo:
        gr.Markdown("""
# Qwen3-TTS Phase 7 - 数据准备与训练提交

本页面只通过 HTTP 调 FastAPI：
1) 收集样本（音频/zip）
2) 生成 train_raw.jsonl
3) 提交训练任务
4) 查询任务状态

说明：ASR 步骤在 MVP 第一版使用占位结果（按文件名生成），可在表格中人工修正文案。
""")

        api_base = gr.Textbox(label="FastAPI Base URL", value=default_api_base)

        with gr.Row():
            upload_files = gr.Files(label="上传音频文件（支持多选）", file_count="multiple", file_types=["audio"])
            upload_archives = gr.Files(label="上传 zip 压缩包（可选）", file_count="multiple", file_types=[".zip"])
            output_name = gr.Textbox(label="输出文件前缀", value="train_raw")
        use_asr_placeholder = gr.Checkbox(
            label="使用 ASR 占位文本（按文件名生成，可编辑）",
            value=True,
        )

        build_rows_btn = gr.Button("1) 通过 HTTP 收集样本并展示")
        samples_table = gr.Dataframe(
            headers=["audio", "text", "asr_text"],
            datatype=["str", "str", "str"],
            row_count=(0, "dynamic"),
            col_count=(3, "fixed"),
            label="样本表（可编辑）",
        )

        save_jsonl_btn = gr.Button("2) 通过 HTTP 生成 train_raw.jsonl")
        jsonl_path = gr.Textbox(label="train_raw.jsonl 路径")
        save_msg = gr.Textbox(label="JSONL 生成结果")

        with gr.Row():
            base_model_id = gr.Number(label="base_model_id", value=1, precision=0)
            speaker_name = gr.Textbox(label="speaker_name", value="speaker_phase7")
            num_epochs = gr.Number(label="num_epochs", value=3, precision=0)
            batch_size = gr.Number(label="batch_size", value=2, precision=0)
            lr = gr.Number(label="lr", value=2e-5)

        submit_train_btn = gr.Button("3) 提交训练任务（HTTP）")
        job_id = gr.Textbox(label="job_id")
        submit_msg = gr.Textbox(label="训练提交结果")

        query_job_btn = gr.Button("4) 查询任务状态（HTTP）")
        job_detail = gr.Textbox(label="任务详情", lines=18)

        build_rows_btn.click(
            fn=_collect_samples,
            inputs=[api_base, upload_files, upload_archives, use_asr_placeholder],
            outputs=[samples_table, save_msg],
        )

        save_jsonl_btn.click(
            fn=_save_train_jsonl,
            inputs=[api_base, output_name, samples_table],
            outputs=[jsonl_path, save_msg],
        )

        submit_train_btn.click(
            fn=_submit_training,
            inputs=[api_base, base_model_id, speaker_name, jsonl_path, num_epochs, batch_size, lr],
            outputs=[job_id, submit_msg],
        )

        query_job_btn.click(
            fn=_query_job,
            inputs=[api_base, job_id],
            outputs=[job_detail],
        )

    return demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-tts-phase7-data-prep",
        description="Launch Phase 7 data-prep GUI that talks to FastAPI over HTTP.",
    )
    parser.add_argument("--api-base", default="http://127.0.0.1:8010", help="FastAPI base URL")
    parser.add_argument("--ip", default="127.0.0.1", help="Gradio bind IP")
    parser.add_argument("--port", type=int, default=8020, help="Gradio bind port")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    demo = create_phase7_data_prep_demo(default_api_base=args.api_base)
    demo.launch(server_name=args.ip, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
