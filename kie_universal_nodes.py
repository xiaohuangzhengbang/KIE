import io
import json
import os
import re
import shutil
import ssl
import subprocess
import tempfile
import threading
import time
import wave
from datetime import datetime

import numpy as np
import requests
import torch
import urllib3
from PIL import Image


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass


KIE_API_HOST = "https://api.kie.ai"
KIE_DOCS_LLM_TXT = "https://docs.kie.ai/llms.txt"
NODE_DIR = os.path.dirname(os.path.abspath(__file__))
PRESET_FILE = os.path.join(NODE_DIR, "kie_model_presets.json")
TASK_FILE = os.path.join(NODE_DIR, "kie_universal_history.json")
task_file_lock = threading.Lock()

TEXT_MODEL_PREFIXES = ("chat/", "claude/", "gemini/", "codex/")

BUILT_IN_PRESETS = [
    {"label": "image: GPT Image-2 - Text to Image", "model": "gpt-image-2-text-to-image", "category": "image", "doc": "https://docs.kie.ai/market/gpt/gpt-image-2-text-to-image.md"},
    {"label": "image: GPT Image 2 - Image To Image", "model": "gpt-image-2-image-to-image", "category": "image", "doc": "https://docs.kie.ai/market/gpt/gpt-image-2-image-to-image.md"},
    {"label": "image: Google - Nano Banana", "model": "google/nano-banana", "category": "image", "doc": "https://docs.kie.ai/market/google/nano-banana.md"},
    {"label": "image: Google - Nano Banana Edit", "model": "google/nano-banana-edit", "category": "image", "doc": "https://docs.kie.ai/market/google/nano-banana-edit.md"},
    {"label": "image: Google - Nano Banana 2", "model": "nano-banana-2", "category": "image", "doc": "https://docs.kie.ai/market/google/nanobanana2.md"},
    {"label": "image: Google - Nano Banana Pro", "model": "nano-banana-pro", "category": "image", "doc": "https://docs.kie.ai/market/google/pro-image-to-image.md"},
    {"label": "image: Seedream5.0 Lite - Text to Image", "model": "seedream/5-lite-text-to-image", "category": "image", "doc": "https://docs.kie.ai/market/seedream/5-lite-text-to-image.md"},
    {"label": "image: Seedream5.0 Lite - Image to Image", "model": "seedream/5-lite-image-to-image", "category": "image", "doc": "https://docs.kie.ai/market/seedream-5-lite-image-to-image.md"},
    {"label": "video: Grok Imagine Text to Video", "model": "grok-imagine/text-to-video", "category": "video", "doc": "https://docs.kie.ai/market/grok-imagine/text-to-video.md"},
    {"label": "video: Grok Imagine Image to Video", "model": "grok-imagine/image-to-video", "category": "video", "doc": "https://docs.kie.ai/market/grok-imagine/image-to-video.md"},
    {"label": "video: Kling 2.6 Text to Video", "model": "kling/text-to-video", "category": "video", "doc": "https://docs.kie.ai/market/kling/text-to-video.md"},
    {"label": "video: Kling 2.6 Image to Video", "model": "kling/image-to-video", "category": "video", "doc": "https://docs.kie.ai/market/kling/image-to-video.md"},
    {"label": "video: Bytedance Seedance 2.0 Fast", "model": "bytedance/seedance-2-fast", "category": "video", "doc": "https://docs.kie.ai/market/bytedance/seedance-2-fast.md"},
    {"label": "video: Wan 2.7 - Text to Video", "model": "wan/2-7-text-to-video", "category": "video", "doc": "https://docs.kie.ai/market/wan/2-7-text-to-video.md"},
    {"label": "video: Wan 2.7 - Image to Video", "model": "wan/2-7-image-to-video", "category": "video", "doc": "https://docs.kie.ai/market/wan/2-7-image-to-video.md"},
    {"label": "video: Topaz - Video Upscale", "model": "topaz/video-upscale", "category": "video", "doc": "https://docs.kie.ai/market/topaz/video-upscale.md"},
    {"label": "audio: elevenlabs/text-to-speech-multilingual-v2", "model": "elevenlabs/text-to-speech-multilingual-v2", "category": "audio", "doc": "https://docs.kie.ai/market/elevenlabs/text-to-speech-multilingual-v2.md"},
    {"label": "audio: elevenlabs/text-to-speech-turbo-2-5", "model": "elevenlabs/text-to-speech-turbo-2-5", "category": "audio", "doc": "https://docs.kie.ai/market/elevenlabs/text-to-speech-turbo-2-5.md"},
    {"label": "audio: elevenlabs/audio-isolation", "model": "elevenlabs/audio-isolation", "category": "audio", "doc": "https://docs.kie.ai/market/elevenlabs/audio-isolation.md"},
]


class KieVideoAdapter:
    def __init__(self, video_path):
        self.video_path = video_path

    def get_dimensions(self):
        return (1280, 720)

    def save_to(self, output_path, **kwargs):
        try:
            if self.video_path and os.path.exists(self.video_path):
                shutil.copyfile(self.video_path, output_path)
                return True
        except Exception:
            pass
        return False


def _get_headers(api_key):
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key.strip()}",
    }


def _get_session():
    session = requests.Session()
    session.trust_env = False
    session.verify = False
    retry = requests.adapters.Retry(total=3, backoff_factor=1)
    adapter = requests.adapters.HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _load_json_file(path, fallback):
    if not os.path.exists(path):
        return fallback
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return fallback


def _write_json_file(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def _load_presets():
    presets = _load_json_file(PRESET_FILE, None)
    if isinstance(presets, list) and presets:
        return presets
    return BUILT_IN_PRESETS


def _preset_labels():
    return [item["label"] for item in _load_presets()] + ["custom: 手动填写 model id"]


def _preset_by_label(label):
    for item in _load_presets():
        if item.get("label") == label:
            return item
    return {}


def _infer_category(text):
    lowered = text.lower()
    if "video" in lowered or "avatar" in lowered or "omnihuman" in lowered or "lip-sync" in lowered:
        return "video"
    if "audio" in lowered or "speech" in lowered or "dialogue" in lowered or "elevenlabs" in lowered:
        return "audio"
    if any(token in lowered for token in ["image", "banana", "flux", "ideogram", "qwen", "recraft", "topaz", "seedream", "imagen", "z-image"]):
        return "image"
    return "media"


def _extract_yaml_block(markdown):
    match = re.search(r"```yaml\s*(.*?)\s*```", markdown, re.S)
    return match.group(1) if match else markdown


def _extract_model_id(yaml_text, fallback):
    model_section = re.search(r"\n\s{16}model:\s*\n(?P<body>.*?)(?:\n\s{16}[A-Za-z_][\w-]*:|\n\s{14}required:|\Z)", yaml_text, re.S)
    body = model_section.group("body") if model_section else yaml_text
    for pattern in [r"\n\s*default:\s*['\"]?([^'\"\n]+)['\"]?", r"\n\s*-\s*['\"]?([^'\"\n]+)['\"]?"]:
        match = re.search(pattern, body)
        if match:
            value = match.group(1).strip()
            if value and not value.startswith("*"):
                return value
    match = re.search(r"model:\s*['\"]?([^'\"\n]+)['\"]?", yaml_text)
    return match.group(1).strip() if match else fallback


def refresh_model_presets():
    session = _get_session()
    llms_text = session.get(KIE_DOCS_LLM_TXT, timeout=30).text
    links = re.findall(
        r"\[([^\]]+)\]\((https://docs\.kie\.ai/market/(?!quickstart|common/get-task-detail)[^)]+\.md)\)",
        llms_text,
    )
    presets = []
    for name, url in links:
        slug = url.split("/market/", 1)[1].removesuffix(".md")
        if slug.startswith(TEXT_MODEL_PREFIXES):
            continue
        lowered = f"{name} {slug}".lower()
        if "chat" in lowered or "language model" in lowered:
            continue
        try:
            markdown = session.get(url, timeout=20).text
            yaml_text = _extract_yaml_block(markdown)
            combined = f"{name} {slug} {yaml_text[:2000]}"
            category = _infer_category(combined)
            model_id = _extract_model_id(yaml_text, slug)
        except Exception:
            category = _infer_category(f"{name} {slug}")
            model_id = slug
        if model_id.startswith(TEXT_MODEL_PREFIXES):
            continue
        presets.append(
            {
                "label": f"{category}: {name.strip()}",
                "model": model_id,
                "category": category,
                "doc": url,
            }
        )

    deduped = []
    seen = set()
    for item in presets:
        key = (item["label"], item["model"])
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    deduped.sort(key=lambda item: (item["category"], item["label"]))
    if deduped:
        _write_json_file(PRESET_FILE, deduped)
    return deduped


def _read_tasks():
    data = _load_json_file(TASK_FILE, {})
    return data if isinstance(data, dict) else {}


def _latest_video_task():
    tasks = _read_tasks()
    candidates = []
    for task_id, record in tasks.items():
        if not isinstance(record, dict):
            continue
        query_type = record.get("query_type")
        if query_type not in {"jobs", "veo"}:
            continue
        timestamp = record.get("updated_at") or record.get("submitted_at") or ""
        candidates.append((timestamp, task_id, record))
    if not candidates:
        raise ValueError("没有找到已提交的视频任务，请先运行 Kie 视频模型异步提交。")
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, task_id, record = candidates[0]
    return task_id, record


def _save_task(task_id, record):
    with task_file_lock:
        tasks = _read_tasks()
        old = tasks.get(task_id, {})
        if not old:
            orders = [
                item.get("submit_order", 0)
                for item in tasks.values()
                if isinstance(item, dict)
            ]
            old["submit_order"] = (max(orders) if orders else 0) + 1
        old.update(record)
        old["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tasks[task_id] = old
        ordered = sorted(
            tasks.items(),
            key=lambda item: item[1].get("updated_at", ""),
            reverse=True,
        )[:200]
        _write_json_file(TASK_FILE, {key: value for key, value in ordered})


def _parse_json(text, default):
    if isinstance(text, (dict, list)):
        return text
    value = (text or "").strip()
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception as exc:
        raise ValueError(f"JSON 解析失败: {exc}")


def _split_url_list(value):
    if not value:
        return []
    text = str(value).replace("\r", "\n").replace(",", "\n").replace("|", "\n")
    return [item.strip() for item in text.split("\n") if item.strip().startswith("http")]


def _tensor_to_temp_file(image_tensor):
    tensor = image_tensor
    if hasattr(tensor, "dim") and tensor.dim() == 4:
        tensor = tensor[0]
    array = tensor.detach().cpu().numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(array)
    if image.mode != "RGB":
        image = image.convert("RGB")
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(temp_file.name, format="PNG")
    temp_file.close()
    return temp_file.name


def _audio_to_temp_wav(audio):
    if not audio:
        return ""
    if isinstance(audio, str) and os.path.exists(audio):
        return audio
    if isinstance(audio, dict):
        for key in ["path", "file", "filename", "audio_path"]:
            path = audio.get(key)
            if isinstance(path, str) and os.path.exists(path):
                return path
        waveform = audio.get("waveform")
        sample_rate = audio.get("sample_rate") or audio.get("sampling_rate") or 44100
        if waveform is not None:
            tensor = waveform.detach().cpu().float() if hasattr(waveform, "detach") else torch.tensor(waveform).float()
            if tensor.dim() == 3:
                tensor = tensor[0]
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            if tensor.dim() != 2:
                raise ValueError(f"音频格式不支持，waveform 形状为 {tuple(tensor.shape)}")
            tensor = tensor.clamp(-1.0, 1.0)
            pcm = (tensor.numpy() * 32767.0).astype(np.int16)
            channels, frames = pcm.shape
            if channels == 1:
                interleaved = pcm[0]
            else:
                interleaved = pcm.T.reshape(frames * channels)
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_file.close()
            with wave.open(temp_file.name, "wb") as wav:
                wav.setnchannels(channels)
                wav.setsampwidth(2)
                wav.setframerate(int(sample_rate))
                wav.writeframes(interleaved.tobytes())
            return temp_file.name
    raise ValueError("音频输入无法保存，请使用 ComfyUI 的加载音频节点输出 AUDIO。")


def _video_to_temp_file(video):
    if not video:
        return ""
    if isinstance(video, str) and os.path.exists(video):
        return video
    if isinstance(video, dict):
        for key in ["path", "file", "filename", "video_path"]:
            path = video.get(key)
            if isinstance(path, str) and os.path.exists(path):
                return path
    for attr in ["video_path", "path", "file", "filename"]:
        path = getattr(video, attr, None)
        if isinstance(path, str) and os.path.exists(path):
            return path
    if hasattr(video, "save_to"):
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_file.close()
        if video.save_to(temp_file.name):
            return temp_file.name
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)
    raise ValueError("视频输入无法保存，请使用 ComfyUI 的加载视频节点输出 VIDEO。")


def _upload_media_input(api_key, media, kind):
    if media is None:
        return "", []
    path = _video_to_temp_file(media) if kind == "video" else _audio_to_temp_wav(media)
    if not path:
        return "", []
    uploaded = _upload_file(api_key, path)
    keep_original = isinstance(media, str) and os.path.abspath(media) == os.path.abspath(path)
    if isinstance(media, dict) and any(isinstance(media.get(key), str) and os.path.abspath(media.get(key)) == os.path.abspath(path) for key in ["path", "file", "filename", "video_path", "audio_path"]):
        keep_original = True
    return uploaded, ([] if keep_original else [path])


def _upload_file(api_key, file_path):
    if not file_path or not os.path.exists(file_path):
        return ""
    upload_name = os.path.basename(file_path)
    with open(file_path, "rb") as file_obj:
        response = requests.post(
            "https://kieai.redpandaai.co/api/file-stream-upload",
            headers={"Authorization": f"Bearer {api_key.strip()}"},
            data={"uploadPath": "comfy", "fileName": upload_name},
            files={"file": (upload_name, file_obj)},
            timeout=120,
            verify=False,
        )
    response.raise_for_status()
    data = response.json()
    if data.get("success"):
        return data.get("data", {}).get("downloadUrl", "")
    raise ValueError(data.get("message") or data.get("msg") or "文件上传失败")


def _prepare_uploaded_urls(api_key, image, image2, image3, image_urls):
    temp_files = []
    urls = []
    try:
        for image_tensor in [image, image2, image3]:
            if image_tensor is None:
                continue
            temp_file = _tensor_to_temp_file(image_tensor)
            temp_files.append(temp_file)
            urls.append(_upload_file(api_key, temp_file))
        urls.extend(_split_url_list(image_urls))
        deduped = []
        seen = set()
        for url in urls:
            if url and url not in seen:
                seen.add(url)
                deduped.append(url)
        return deduped, temp_files
    except Exception:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        raise


def _assign_media_urls(input_payload, urls, field_name, append_to_existing):
    if not urls:
        return input_payload
    payload = dict(input_payload)
    field = (field_name or "").strip()
    if not field or field == "不自动写入":
        return payload
    existing = payload.get(field)
    if append_to_existing and isinstance(existing, list):
        payload[field] = existing + urls
    elif field in {"image_url", "video_url", "audio_url", "file_url"}:
        payload[field] = urls[0]
    else:
        payload[field] = urls
    return payload


def _submit_task(api_key, model_id, input_payload, callback_url=""):
    payload = {"model": model_id, "input": input_payload}
    if callback_url.strip():
        payload["callBackUrl"] = callback_url.strip()
    session = _get_session()
    response = session.post(
        f"{KIE_API_HOST}/api/v1/jobs/createTask",
        headers=_get_headers(api_key),
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("code") != 200:
        raise ValueError(data.get("msg") or "提交任务失败")
    task_id = data.get("data", {}).get("taskId")
    if not task_id:
        raise ValueError("接口没有返回 taskId")
    return task_id, data


def _query_task(api_key, task_id):
    session = _get_session()
    response = session.get(
        f"{KIE_API_HOST}/api/v1/jobs/recordInfo",
        headers=_get_headers(api_key),
        params={"taskId": task_id},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("code") != 200:
        raise ValueError(data.get("msg") or "查询任务失败")
    return data.get("data", {}), data


def _parse_nested_json(value):
    if isinstance(value, str):
        text = value.strip()
        if text and text[0] in "[{":
            try:
                return json.loads(text)
            except Exception:
                return value
    return value


def _collect_urls(value, results):
    parsed = _parse_nested_json(value)
    if isinstance(parsed, str):
        if parsed.startswith("http://") or parsed.startswith("https://"):
            results.append(parsed)
        return
    if isinstance(parsed, list):
        for item in parsed:
            _collect_urls(item, results)
        return
    if isinstance(parsed, dict):
        for item in parsed.values():
            _collect_urls(item, results)


def _extract_urls(task_data):
    urls = []
    _collect_urls(task_data, urls)
    deduped = []
    seen = set()
    for url in urls:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped


def _task_state(task_data):
    success_flag = task_data.get("successFlag")
    if success_flag == 1:
        return "success"
    if success_flag in {2, 3}:
        return "fail"
    state = str(task_data.get("state") or task_data.get("status") or "").strip().lower()
    if state in {"success", "succeeded", "completed", "done"}:
        return "success"
    if state in {"fail", "failed", "error", "canceled", "cancelled"}:
        return "fail"
    return state or "waiting"


def _task_error(task_data):
    return task_data.get("failMsg") or task_data.get("errorMessage") or task_data.get("message") or ""


def _download(url, save_path):
    if shutil.which("curl"):
        try:
            subprocess.call(
                [shutil.which("curl"), "-kL", "--retry", "3", "-o", save_path, url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                return True
        except Exception:
            pass
    session = _get_session()
    with session.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
    return os.path.exists(save_path) and os.path.getsize(save_path) > 0


def _extension_from_url(url, fallback):
    lowered = url.lower().split("?", 1)[0]
    for ext in [".mp4", ".mov", ".webm", ".png", ".jpg", ".jpeg", ".webp", ".wav", ".mp3", ".m4a", ".json"]:
        if lowered.endswith(ext):
            return ext
    if "video" in lowered:
        return ".mp4"
    if "audio" in lowered:
        return ".mp3"
    if "image" in lowered or "img" in lowered:
        return ".png"
    return fallback


def _load_image_tensor(path):
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array)[None,]


def _blank_image():
    return torch.zeros((1, 64, 64, 3), dtype=torch.float32)


def _force_run_token(*args, **kwargs):
    return float("NaN")


VIDEO_ASYNC_MODELS = [
    "veo3.1-lite",
    "veo3.1-fast",
    "veo3.1-quality",
    "grok-imagine",
    "grok-imagine-video-1.5-preview",
    "seedance-2",
    "seedance-2-fast",
    "seedance-2-mini",
    "seedance-1.5-pro",
]

VIDEO_ASPECT_RATIOS = ["Auto", "16:9", "9:16", "1:1", "2:3", "3:2"]
VIDEO_RESOLUTIONS = ["720p", "1080p", "4K"]
SEEDANCE_MODES = ["自动", "首帧图生视频", "首尾帧图生视频", "多模态参考视频"]

VEO_MODEL_MAP = {
    "veo3.1-lite": "veo3_lite",
    "veo3.1-fast": "veo3_fast",
    "veo3.1-quality": "veo3",
}

SEEDANCE_MODEL_MAP = {
    "seedance-2": "bytedance/seedance-2",
    "seedance-2-fast": "bytedance/seedance-2-fast",
    "seedance-2-mini": "bytedance/seedance-2-mini",
    "seedance-1.5-pro": "bytedance/seedance-1.5-pro",
}


def _merge_extra_json(payload, extra_json):
    extra = _parse_json(extra_json, {})
    if not isinstance(extra, dict):
        raise ValueError("附加JSON必须是对象")
    merged = dict(payload)
    merged.update(extra)
    return merged


def _normalize_video_aspect_ratio(value, default="16:9"):
    aspect_ratio = (value or default).strip()
    if aspect_ratio == "Auto":
        return default
    return aspect_ratio


def _normalize_grok_aspect_ratio(value):
    aspect_ratio = _normalize_video_aspect_ratio(value, "2:3")
    return aspect_ratio if aspect_ratio in {"1:1", "16:9", "9:16", "2:3", "3:2"} else "2:3"


def _normalize_veo_aspect_ratio(value):
    aspect_ratio = _normalize_video_aspect_ratio(value, "16:9")
    if aspect_ratio in {"1:1", "3:2"}:
        return "16:9"
    if aspect_ratio == "2:3":
        return "9:16"
    return aspect_ratio if aspect_ratio in {"16:9", "9:16"} else "16:9"


def _normalize_veo_resolution(value):
    resolution = (value or "720p").strip()
    return {"4K": "4k", "1080p": "1080p", "720p": "720p"}.get(resolution, "720p")


def _normalize_duration(value, default=10, minimum=1, maximum=30):
    try:
        duration = int(value)
    except Exception:
        duration = default
    return max(minimum, min(maximum, duration))


def _normalize_seed(value):
    try:
        seed = int(value)
    except Exception:
        return None
    return seed if 10000 <= seed <= 99999 else None


def _guess_reference_urls(media_urls):
    video_urls = []
    audio_urls = []
    other_urls = []
    for url in _split_url_list(media_urls):
        lowered = url.lower().split("?", 1)[0]
        if lowered.endswith((".mp4", ".mov", ".webm", ".m4v")):
            video_urls.append(url)
        elif lowered.endswith((".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg")):
            audio_urls.append(url)
        else:
            other_urls.append(url)
    return video_urls, audio_urls, other_urls


def _ensure_max_items(model_name, field_name, values, max_count):
    count = len([item for item in values if item])
    if count > max_count:
        raise ValueError(f"{model_name} 的{field_name}最多支持 {max_count} 个，当前传入 {count} 个。请删除多余输入后重试。")


def _ensure_no_items(model_name, field_name, values):
    count = len([item for item in values if item])
    if count:
        raise ValueError(f"{model_name} 不支持{field_name}，当前传入 {count} 个。请删除这些输入或切换到支持该输入的模型/模式。")


def _ensure_exact_items(model_name, field_name, values, exact_count):
    count = len([item for item in values if item])
    if count != exact_count:
        raise ValueError(f"{model_name} 的{field_name}需要 {exact_count} 个，当前传入 {count} 个。")


def _seedance_effective_mode(seedance_mode, image_urls, reference_video_urls, reference_audio_urls):
    if seedance_mode != "自动":
        return seedance_mode
    if reference_video_urls or reference_audio_urls or len(image_urls) > 2:
        return "多模态参考视频"
    if len(image_urls) == 2:
        return "首尾帧图生视频"
    if len(image_urls) == 1:
        return "首帧图生视频"
    return "文生视频"


def _build_veo_payload(model_name, prompt, aspect_ratio, resolution, image_urls, seed, watermark, ref_mode):
    image_urls = [url for url in image_urls if url]
    generation_type = "TEXT_2_VIDEO"
    if image_urls:
        generation_type = "REFERENCE_2_VIDEO" if ref_mode else "FIRST_AND_LAST_FRAMES_2_VIDEO"

    veo_aspect_ratio = _normalize_veo_aspect_ratio(aspect_ratio)
    if generation_type == "REFERENCE_2_VIDEO":
        _ensure_max_items(model_name, "参考图片", image_urls, 3)
        raw_aspect_ratio = (aspect_ratio or "").strip()
        if raw_aspect_ratio not in {"Auto", "16:9", "9:16"}:
            raise ValueError(f"{model_name} 的 Veo参考模式 只支持 16:9 或 9:16，当前选择 {raw_aspect_ratio}。")
    elif generation_type == "FIRST_AND_LAST_FRAMES_2_VIDEO":
        _ensure_max_items(model_name, "首尾帧图片", image_urls, 2)
    elif not prompt.strip():
        raise ValueError(f"{model_name} 文生视频需要提示词。")

    payload = {
        "prompt": prompt,
        "model": VEO_MODEL_MAP[model_name],
        "aspect_ratio": veo_aspect_ratio,
        "enableTranslation": True,
        "generationType": generation_type,
        "resolution": _normalize_veo_resolution(resolution),
    }
    if image_urls:
        payload["imageUrls"] = image_urls
    normalized_seed = _normalize_seed(seed)
    if normalized_seed is not None:
        payload["seeds"] = normalized_seed
    if watermark.strip():
        payload["watermark"] = watermark.strip()
    return payload


def _build_grok_payload(model_name, prompt, aspect_ratio, resolution, duration, image_urls, task_id, task_index, nsfw_checker):
    image_urls = [url for url in image_urls if url]
    _ensure_max_items(model_name, "图片", image_urls, 1)
    payload = {
        "mode": "normal",
        "duration": _normalize_duration(duration, 6, 6, 30),
        "resolution": resolution,
        "aspect_ratio": _normalize_grok_aspect_ratio(aspect_ratio),
        "nsfw_checker": bool(nsfw_checker),
    }
    if prompt.strip():
        payload["prompt"] = prompt
    if model_name == "grok-imagine-video-1.5-preview":
        if image_urls:
            payload["image_urls"] = image_urls
        elif not prompt.strip():
            raise ValueError("grok-imagine-video-1.5-preview 需要提示词或 1 张图片。")
        return "grok-imagine-video-1-5-preview", payload
    if image_urls:
        if task_id.strip():
            payload["task_id"] = task_id.strip()
            payload["index"] = max(0, min(5, int(task_index or 0)))
        else:
            payload["image_urls"] = image_urls
        payload["duration"] = str(payload["duration"])
        return "grok-imagine/image-to-video", payload
    if not prompt.strip():
        raise ValueError("Grok 文生视频需要提示词")
    return "grok-imagine/text-to-video", payload


def _build_seedance_payload(
    model_name,
    prompt,
    aspect_ratio,
    resolution,
    duration,
    image_urls,
    reference_video_urls,
    reference_audio_urls,
    seedance_mode,
    generate_audio,
    return_last_frame,
    web_search,
):
    image_urls = [url for url in image_urls if url]
    reference_video_urls = [url for url in reference_video_urls if url]
    reference_audio_urls = [url for url in reference_audio_urls if url]
    payload = {
        "prompt": prompt,
        "return_last_frame": bool(return_last_frame),
        "generate_audio": bool(generate_audio),
        "resolution": "4k" if resolution == "4K" else resolution,
        "aspect_ratio": _normalize_video_aspect_ratio(aspect_ratio, "16:9"),
        "duration": _normalize_duration(duration, 10, 1, 30),
        "web_search": bool(web_search),
    }
    mode = _seedance_effective_mode(seedance_mode, image_urls, reference_video_urls, reference_audio_urls)

    if mode == "首帧图生视频":
        _ensure_exact_items(model_name, "首帧图片", image_urls, 1)
        _ensure_no_items(model_name, "视频", reference_video_urls)
        _ensure_no_items(model_name, "音频", reference_audio_urls)
        payload["first_frame_url"] = image_urls[0]
    elif mode == "首尾帧图生视频":
        _ensure_exact_items(model_name, "首尾帧图片", image_urls, 2)
        _ensure_no_items(model_name, "视频", reference_video_urls)
        _ensure_no_items(model_name, "音频", reference_audio_urls)
        payload["first_frame_url"] = image_urls[0]
        payload["last_frame_url"] = image_urls[1]
    elif mode == "多模态参考视频":
        _ensure_max_items(model_name, "参考图片", image_urls, 9)
        _ensure_max_items(model_name, "视频", reference_video_urls, 1)
        _ensure_max_items(model_name, "音频", reference_audio_urls, 1)
        if not image_urls and not reference_video_urls and not reference_audio_urls:
            raise ValueError("Seedance 多模态参考视频至少需要 1 个图片、视频或音频。")
        if image_urls:
            payload["reference_image_urls"] = image_urls
        if reference_video_urls:
            payload["reference_video_urls"] = reference_video_urls
        if reference_audio_urls:
            payload["reference_audio_urls"] = reference_audio_urls
    elif not prompt.strip():
        raise ValueError("Seedance 文生视频需要提示词")
    else:
        _ensure_no_items(model_name, "图片", image_urls)
        _ensure_no_items(model_name, "视频", reference_video_urls)
        _ensure_no_items(model_name, "音频", reference_audio_urls)

    return SEEDANCE_MODEL_MAP[model_name], payload


def _submit_veo_task(api_key, input_payload, callback_url=""):
    payload = dict(input_payload)
    if callback_url.strip():
        payload["callBackUrl"] = callback_url.strip()
    session = _get_session()
    response = session.post(
        f"{KIE_API_HOST}/api/v1/veo/generate",
        headers=_get_headers(api_key),
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("code") != 200:
        raise ValueError(data.get("msg") or "Veo 提交任务失败")
    task_id = data.get("data", {}).get("taskId")
    if not task_id:
        raise ValueError("Veo 接口没有返回 taskId")
    return task_id, data


def _query_veo_task(api_key, task_id):
    session = _get_session()
    response = session.get(
        f"{KIE_API_HOST}/api/v1/veo/record-info",
        headers=_get_headers(api_key),
        params={"taskId": task_id},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("code") != 200:
        raise ValueError(data.get("msg") or "Veo 查询任务失败")
    return data.get("data", {}), data


def _veo_task_state(task_data):
    success_flag = task_data.get("successFlag")
    if success_flag == 1:
        return "success"
    if success_flag in {2, 3}:
        return "fail"
    return str(task_data.get("status") or task_data.get("state") or "waiting").lower()


def _veo_result_urls(task_data):
    response_data = task_data.get("response") or {}
    urls = []
    for key in ["fullResultUrls", "resultUrls", "originUrls"]:
        value = response_data.get(key)
        if isinstance(value, list):
            urls.extend(value)
    if not urls:
        urls = _extract_urls(task_data)
    return list(dict.fromkeys(urls))


def _query_saved_video_task(api_key, task_id, record):
    query_type = record.get("query_type") or "jobs"
    if query_type == "veo":
        task_data, raw = _query_veo_task(api_key, task_id)
        state = _veo_task_state(task_data)
        result_urls = _veo_result_urls(task_data)
        error = task_data.get("errorMessage") or task_data.get("failMsg") or ""
    else:
        task_data, raw = _query_task(api_key, task_id)
        state = _task_state(task_data)
        result_urls = _extract_urls(task_data)
        error = _task_error(task_data)
    _save_task(
        task_id,
        {
            "status": state,
            "result_urls": result_urls,
            "error": error,
            "query_type": query_type,
        },
    )
    return state, result_urls, raw


class KieRefreshModelPresets:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"刷新": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("报告",)
    FUNCTION = "refresh"
    CATEGORY = "KieAI/通用"

    def refresh(self, 刷新=True):
        if not 刷新:
            return ("已跳过刷新",)
        presets = refresh_model_presets()
        counts = {}
        for item in presets:
            counts[item.get("category", "media")] = counts.get(item.get("category", "media"), 0) + 1
        detail = ", ".join(f"{key}: {value}" for key, value in sorted(counts.items()))
        return (f"已刷新 {len(presets)} 个 Kie 媒体模型预设（不含聊天/文本模型）：{detail}",)


class KieUniversalSubmit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "API密钥": ("STRING", {"multiline": False, "default": ""}),
                "模型预设": (_preset_labels(), {"default": _preset_labels()[0]}),
                "自定义model": ("STRING", {"multiline": False, "default": ""}),
                "输入JSON": ("STRING", {"multiline": True, "default": "{\n  \"prompt\": \"A beautiful image.\"\n}"}),
                "上传URL字段": (["不自动写入", "image_urls", "image_url", "video_urls", "video_url", "audio_url", "file_url"], {"default": "image_urls"}),
                "合并已有URL数组": ("BOOLEAN", {"default": True}),
                "等待完成": ("BOOLEAN", {"default": False}),
                "轮询秒数": ("INT", {"default": 180, "min": 0, "max": 3600, "step": 5}),
            },
            "optional": {
                "图片": ("IMAGE",),
                "图片2": ("IMAGE",),
                "图片3": ("IMAGE",),
                "媒体URL": ("STRING", {"multiline": True, "default": ""}),
                "回调URL": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("任务ID", "状态", "结果URL", "原始JSON")
    FUNCTION = "submit"
    CATEGORY = "KieAI/通用"
    IS_CHANGED = _force_run_token

    def submit(
        self,
        API密钥,
        模型预设,
        自定义model,
        输入JSON,
        上传URL字段,
        合并已有URL数组,
        等待完成,
        轮询秒数,
        图片=None,
        图片2=None,
        图片3=None,
        媒体URL="",
        回调URL="",
    ):
        api_key = API密钥.strip()
        if not api_key:
            raise ValueError("缺少 API 密钥")
        preset = _preset_by_label(模型预设)
        model_id = 自定义model.strip() or preset.get("model", "")
        if not model_id:
            raise ValueError("缺少 model id")
        input_payload = _parse_json(输入JSON, {})
        urls, temp_files = _prepare_uploaded_urls(api_key, 图片, 图片2, 图片3, 媒体URL)
        try:
            input_payload = _assign_media_urls(input_payload, urls, 上传URL字段, 合并已有URL数组)
            task_id, submit_data = _submit_task(api_key, model_id, input_payload, 回调URL)
            _save_task(
                task_id,
                {
                    "model": model_id,
                    "preset": 模型预设,
                    "status": "submitted",
                    "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "input": input_payload,
                },
            )

            if not 等待完成:
                return (task_id, "submitted", "", json.dumps(submit_data, ensure_ascii=False))

            deadline = time.time() + max(0, int(轮询秒数))
            last_data = submit_data
            while time.time() <= deadline:
                time.sleep(5)
                task_data, raw = _query_task(api_key, task_id)
                last_data = raw
                state = _task_state(task_data)
                result_urls = _extract_urls(task_data)
                _save_task(
                    task_id,
                    {
                        "model": model_id,
                        "preset": 模型预设,
                        "status": state,
                        "result_urls": result_urls,
                        "error": _task_error(task_data),
                    },
                )
                if state == "success":
                    return (task_id, state, "\n".join(result_urls), json.dumps(raw, ensure_ascii=False))
                if state == "fail":
                    return (task_id, state, "\n".join(result_urls), json.dumps(raw, ensure_ascii=False))
            return (task_id, "timeout", "", json.dumps(last_data, ensure_ascii=False))
        finally:
            for temp_file in temp_files:
                if temp_file and os.path.exists(temp_file):
                    os.remove(temp_file)


class KieUniversalQuery:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "API密钥": ("STRING", {"multiline": False, "default": ""}),
                "任务ID": ("STRING", {"multiline": False, "default": "", "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("状态", "结果URL", "原始JSON")
    FUNCTION = "query"
    CATEGORY = "KieAI/通用"
    IS_CHANGED = _force_run_token

    def query(self, API密钥, 任务ID):
        api_key = API密钥.strip()
        task_id = 任务ID.strip()
        if not api_key:
            raise ValueError("缺少 API 密钥")
        if not task_id:
            raise ValueError("缺少任务ID")
        task_data, raw = _query_task(api_key, task_id)
        state = _task_state(task_data)
        result_urls = _extract_urls(task_data)
        _save_task(
            task_id,
            {
                "status": state,
                "result_urls": result_urls,
                "error": _task_error(task_data),
            },
        )
        return (state, "\n".join(result_urls), json.dumps(raw, ensure_ascii=False))


class KieUniversalDownload:
    @classmethod
    def INPUT_TYPES(cls):
        return {}

    RETURN_TYPES = ("IMAGE", "VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("图像", "视频", "文件路径", "下载URL")
    FUNCTION = "download"
    CATEGORY = "KieAI/通用"
    IS_CHANGED = _force_run_token

    def download(self):
        video, report = KieVideoResultDownload().download()
        return (_blank_image(), video, video.video_path, report)


def _select_video_download_candidate(tasks):
    downloaded_urls = {
        url
        for task in tasks.values()
        if isinstance(task, dict) and task.get("status") == "downloaded"
        for url in task.get("result_urls", [])
        if isinstance(url, str) and url
    }
    candidates = [
        (task_id, task)
        for task_id, task in tasks.items()
        if isinstance(task, dict)
        and task.get("query_type") in {"jobs", "veo"}
        and task.get("status") == "success"
        and task.get("result_urls")
        and not task.get("video_path")
        and not task.get("downloading")
        and task["result_urls"][0] not in downloaded_urls
    ]
    candidates.sort(
        key=lambda item: (
            item[1].get("submit_order", -1),
            item[1].get("submitted_at", ""),
            item[1].get("updated_at", ""),
            item[0],
        ),
        reverse=True,
    )
    return candidates[0] if candidates else None


class KieVideoResultDownload:
    @classmethod
    def INPUT_TYPES(cls):
        return {}

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("视频", "报告")
    FUNCTION = "download"
    CATEGORY = "KieAI/视频"
    IS_CHANGED = _force_run_token

    def download(self):
        with task_file_lock:
            tasks = _read_tasks()
            target = _select_video_download_candidate(tasks)
            if target:
                task_id, task = target
                task["downloading"] = True
                task["download_started_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                _write_json_file(TASK_FILE, tasks)

        if not target:
            failed = [
                (task_id, task)
                for task_id, task in tasks.items()
                if isinstance(task, dict)
                and task.get("query_type") in {"jobs", "veo"}
                and task.get("status") == "fail"
            ]
            failed.sort(
                key=lambda item: (
                    item[1].get("submit_order", -1),
                    item[1].get("updated_at", ""),
                ),
                reverse=True,
            )
            message = "没有新的成功视频可下载。请先运行“查询全部视频任务”。"
            if failed:
                task_id, task = failed[0]
                reason = task.get("error") or task.get("last_query_error") or "未知错误"
                message += f" 最新失败任务: {task_id}，原因: {reason}"
            raise ValueError(message)

        task_id, task = target
        url = task["result_urls"][0]
        try:
            import folder_paths

            output_dir = folder_paths.get_output_directory()
        except Exception:
            output_dir = NODE_DIR
        model = re.sub(r"[^A-Za-z0-9._-]+", "_", task.get("model", "video"))
        safe_task_id = re.sub(r"[^A-Za-z0-9._-]+", "_", task_id)
        extension = _extension_from_url(url, ".mp4")
        if extension not in {".mp4", ".mov", ".webm", ".mkv"}:
            extension = ".mp4"
        filename = f"kie_{model}_{safe_task_id}{extension}"
        output_path = os.path.join(output_dir, filename)

        try:
            if not _download(url, output_path):
                raise RuntimeError("下载接口没有返回有效文件")
        except Exception as exc:
            _save_task(
                task_id,
                {
                    "downloading": False,
                    "download_error": str(exc),
                },
            )
            return (KieVideoAdapter(""), f"下载失败 | {task_id} | 原因: {exc}")

        _save_task(
            task_id,
            {
                "status": "downloaded",
                "video_path": output_path,
                "downloading": False,
                "download_error": "",
                "downloaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        return (KieVideoAdapter(output_path), f"下载成功 | {task_id} | {filename}")


class KieVideoSeriesAsyncSubmit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "API密钥": ("STRING", {"multiline": False, "default": ""}),
                "模型": (VIDEO_ASYNC_MODELS, {"default": "veo3.1-fast"}),
                "提示词": ("STRING", {"multiline": True, "default": ""}),
                "画幅": (VIDEO_ASPECT_RATIOS, {"default": "16:9"}),
                "分辨率": (VIDEO_RESOLUTIONS, {"default": "720p"}),
                "时长秒": ("INT", {"default": 10, "min": 1, "max": 30, "step": 1}),
                "Seedance模式": (SEEDANCE_MODES, {"default": "自动"}),
                "Veo参考模式": ("BOOLEAN", {"default": False}),
                "生成音频": ("BOOLEAN", {"default": False}),
                "返回尾帧": ("BOOLEAN", {"default": False}),
                "联网增强": ("BOOLEAN", {"default": False}),
                "NSFW检查": ("BOOLEAN", {"default": True}),
                "种子": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}),
                "附加JSON": ("STRING", {"multiline": True, "default": "{}"}),
            },
            "optional": {
                "图片": ("IMAGE",),
                "图片2": ("IMAGE",),
                "图片3": ("IMAGE",),
                "媒体URL": ("STRING", {"multiline": True, "default": ""}),
                "参考视频URL": ("STRING", {"multiline": True, "default": ""}),
                "参考音频URL": ("STRING", {"multiline": True, "default": ""}),
                "Grok任务ID": ("STRING", {"multiline": False, "default": ""}),
                "Grok任务图片序号": ("INT", {"default": 0, "min": 0, "max": 5, "step": 1}),
                "水印": ("STRING", {"multiline": False, "default": ""}),
                "回调URL": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("任务ID", "查询类型", "请求JSON", "原始JSON")
    FUNCTION = "submit"
    CATEGORY = "KieAI/视频"

    def submit(
        self,
        API密钥,
        模型,
        提示词,
        画幅,
        分辨率,
        时长秒,
        Seedance模式,
        Veo参考模式,
        生成音频,
        返回尾帧,
        联网增强,
        NSFW检查,
        种子,
        附加JSON,
        图片=None,
        图片2=None,
        图片3=None,
        媒体URL="",
        参考视频URL="",
        参考音频URL="",
        Grok任务ID="",
        Grok任务图片序号=0,
        水印="",
        回调URL="",
    ):
        api_key = API密钥.strip()
        if not api_key:
            raise ValueError("缺少 API 密钥")

        media_video_urls, media_audio_urls, media_other_urls = _guess_reference_urls(媒体URL)
        image_urls, temp_files = _prepare_uploaded_urls(api_key, 图片, 图片2, 图片3, "\n".join(media_other_urls))
        try:
            reference_video_urls = _split_url_list(参考视频URL) + media_video_urls
            reference_audio_urls = _split_url_list(参考音频URL) + media_audio_urls

            query_type = "jobs"
            if 模型 in VEO_MODEL_MAP:
                _ensure_no_items(模型, "参考视频URL", reference_video_urls)
                _ensure_no_items(模型, "参考音频URL", reference_audio_urls)
                input_payload = _build_veo_payload(模型, 提示词, 画幅, 分辨率, image_urls, 种子, 水印, Veo参考模式)
                input_payload = _merge_extra_json(input_payload, 附加JSON)
                task_id, raw = _submit_veo_task(api_key, input_payload, 回调URL)
                query_type = "veo"
                request_payload = dict(input_payload)
            elif 模型.startswith("grok"):
                _ensure_no_items(模型, "参考视频URL", reference_video_urls)
                _ensure_no_items(模型, "参考音频URL", reference_audio_urls)
                model_id, input_payload = _build_grok_payload(
                    模型,
                    提示词,
                    画幅,
                    分辨率,
                    时长秒,
                    image_urls,
                    Grok任务ID,
                    Grok任务图片序号,
                    NSFW检查,
                )
                input_payload = _merge_extra_json(input_payload, 附加JSON)
                task_id, raw = _submit_task(api_key, model_id, input_payload, 回调URL)
                request_payload = {"model": model_id, "input": input_payload}
            elif 模型 in SEEDANCE_MODEL_MAP:
                model_id, input_payload = _build_seedance_payload(
                    模型,
                    提示词,
                    画幅,
                    分辨率,
                    时长秒,
                    image_urls,
                    reference_video_urls,
                    reference_audio_urls,
                    Seedance模式,
                    生成音频,
                    返回尾帧,
                    联网增强,
                )
                input_payload = _merge_extra_json(input_payload, 附加JSON)
                task_id, raw = _submit_task(api_key, model_id, input_payload, 回调URL)
                request_payload = {"model": model_id, "input": input_payload}
            else:
                raise ValueError(f"不支持的模型: {模型}")

            _save_task(
                task_id,
                {
                    "model": 模型,
                    "status": "submitted",
                    "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "input": request_payload,
                    "query_type": query_type,
                },
            )
            return (task_id, query_type, json.dumps(request_payload, ensure_ascii=False), json.dumps(raw, ensure_ascii=False))
        finally:
            for temp_file in temp_files:
                if temp_file and os.path.exists(temp_file):
                    os.remove(temp_file)


def _prepare_uploaded_url_list(api_key, image_tensors, image_url_text=""):
    temp_files = []
    urls = []
    try:
        for image_tensor in image_tensors:
            if image_tensor is None:
                continue
            temp_file = _tensor_to_temp_file(image_tensor)
            temp_files.append(temp_file)
            urls.append(_upload_file(api_key, temp_file))
        urls.extend(_split_url_list(image_url_text))
        deduped = []
        seen = set()
        for url in urls:
            if url and url not in seen:
                deduped.append(url)
                seen.add(url)
        return deduped, temp_files
    except Exception:
        for temp_file in temp_files:
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
        raise


def _save_video_submit_record(task_id, model_name, request_payload, query_type):
    _save_task(
        task_id,
        {
            "model": model_name,
            "status": "submitted",
            "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input": request_payload,
            "query_type": query_type,
        },
    )


class KieVeoAsyncSubmit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "API密钥": ("STRING", {"multiline": False, "default": ""}),
                "模型": (["veo3.1-lite", "veo3.1-fast", "veo3.1-quality"], {"default": "veo3.1-fast"}),
                "模式": (["文生视频", "首尾帧图生视频", "参考图生视频"], {"default": "文生视频"}),
                "提示词": ("STRING", {"multiline": True, "default": ""}),
                "画幅": (["16:9", "9:16"], {"default": "16:9"}),
                "分辨率": (VIDEO_RESOLUTIONS, {"default": "720p"}),
            },
            "optional": {
                "图片1": ("IMAGE",),
                "图片2": ("IMAGE",),
                "图片3": ("IMAGE",),
                "图片URL": ("STRING", {"multiline": True, "default": ""}),
                "种子": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}),
                "水印": ("STRING", {"multiline": False, "default": ""}),
                "回调URL": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("任务ID", "查询类型", "请求JSON", "原始JSON")
    FUNCTION = "submit"
    CATEGORY = "KieAI/视频"

    def submit(self, API密钥, 模型, 模式, 提示词, 画幅, 分辨率, 图片1=None, 图片2=None, 图片3=None, 图片URL="", 种子=0, 水印="", 回调URL=""):
        api_key = API密钥.strip()
        if not api_key:
            raise ValueError("缺少 API 密钥")
        image_urls, temp_files = _prepare_uploaded_url_list(api_key, [图片1, 图片2, 图片3], 图片URL)
        try:
            if 模式 == "文生视频":
                _ensure_no_items(模型, "图片", image_urls)
                ref_mode = False
            elif 模式 == "首尾帧图生视频":
                _ensure_max_items(模型, "首尾帧图片", image_urls, 2)
                if not image_urls:
                    raise ValueError(f"{模型} 首尾帧图生视频需要 1-2 张图片。")
                ref_mode = False
            else:
                _ensure_max_items(模型, "参考图片", image_urls, 3)
                if not image_urls:
                    raise ValueError(f"{模型} 参考图生视频需要 1-3 张图片。")
                ref_mode = True

            input_payload = _build_veo_payload(模型, 提示词, 画幅, 分辨率, image_urls, 种子, 水印, ref_mode)
            task_id, raw = _submit_veo_task(api_key, input_payload, 回调URL)
            _save_video_submit_record(task_id, 模型, input_payload, "veo")
            return (task_id, "veo", json.dumps(input_payload, ensure_ascii=False), json.dumps(raw, ensure_ascii=False))
        finally:
            for temp_file in temp_files:
                if temp_file and os.path.exists(temp_file):
                    os.remove(temp_file)


class KieGrokAsyncSubmit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "API密钥": ("STRING", {"multiline": False, "default": ""}),
                "模型": (["grok-imagine", "grok-imagine-video-1.5-preview"], {"default": "grok-imagine"}),
                "提示词": ("STRING", {"multiline": True, "default": ""}),
                "画幅": (VIDEO_ASPECT_RATIOS, {"default": "16:9"}),
                "分辨率": (["480p", "720p", "1080p"], {"default": "720p"}),
                "时长秒": ("INT", {"default": 6, "min": 6, "max": 30, "step": 1}),
            },
            "optional": {
                "图片": ("IMAGE",),
                "图片URL": ("STRING", {"multiline": False, "default": ""}),
                "任务ID": ("STRING", {"multiline": False, "default": ""}),
                "任务图片序号": ("INT", {"default": 0, "min": 0, "max": 5, "step": 1}),
                "NSFW检查": ("BOOLEAN", {"default": True}),
                "回调URL": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("任务ID", "查询类型", "请求JSON", "原始JSON")
    FUNCTION = "submit"
    CATEGORY = "KieAI/视频"

    def submit(self, API密钥, 模型, 提示词, 画幅, 分辨率, 时长秒, 图片=None, 图片URL="", 任务ID="", 任务图片序号=0, NSFW检查=True, 回调URL=""):
        api_key = API密钥.strip()
        if not api_key:
            raise ValueError("缺少 API 密钥")
        image_urls, temp_files = _prepare_uploaded_url_list(api_key, [图片], 图片URL)
        try:
            model_id, input_payload = _build_grok_payload(模型, 提示词, 画幅, 分辨率, 时长秒, image_urls, 任务ID, 任务图片序号, NSFW检查)
            task_id, raw = _submit_task(api_key, model_id, input_payload, 回调URL)
            request_payload = {"model": model_id, "input": input_payload}
            _save_video_submit_record(task_id, 模型, request_payload, "jobs")
            return (task_id, "jobs", json.dumps(request_payload, ensure_ascii=False), json.dumps(raw, ensure_ascii=False))
        finally:
            for temp_file in temp_files:
                if temp_file and os.path.exists(temp_file):
                    os.remove(temp_file)


class KieSeedanceAsyncSubmit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "API密钥": ("STRING", {"multiline": False, "default": ""}),
                "模型": (["seedance-2", "seedance-2-fast", "seedance-2-mini", "seedance-1.5-pro"], {"default": "seedance-2"}),
                "模式": (["自动", "文生视频", "首帧图生视频", "首尾帧图生视频", "多模态参考视频"], {"default": "自动"}),
                "提示词": ("STRING", {"multiline": True, "default": ""}),
                "画幅": (["16:9", "9:16", "1:1", "4:3", "3:4"], {"default": "16:9"}),
                "分辨率": (VIDEO_RESOLUTIONS, {"default": "720p"}),
                "时长秒": ("INT", {"default": 10, "min": 1, "max": 30, "step": 1}),
                "生成音频": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "图片1": ("IMAGE",),
                "图片2": ("IMAGE",),
                "图片3": ("IMAGE",),
                "图片4": ("IMAGE",),
                "图片5": ("IMAGE",),
                "图片6": ("IMAGE",),
                "图片7": ("IMAGE",),
                "图片8": ("IMAGE",),
                "图片9": ("IMAGE",),
                "图片URL": ("STRING", {"multiline": True, "default": ""}),
                "视频URL": ("STRING", {"multiline": False, "default": ""}),
                "音频URL": ("STRING", {"multiline": False, "default": ""}),
                "返回尾帧": ("BOOLEAN", {"default": False}),
                "联网增强": ("BOOLEAN", {"default": False}),
                "回调URL": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("任务ID", "查询类型", "请求JSON", "原始JSON")
    FUNCTION = "submit"
    CATEGORY = "KieAI/视频"

    def submit(
        self,
        API密钥,
        模型,
        模式,
        提示词,
        画幅,
        分辨率,
        时长秒,
        生成音频,
        图片1=None,
        图片2=None,
        图片3=None,
        图片4=None,
        图片5=None,
        图片6=None,
        图片7=None,
        图片8=None,
        图片9=None,
        图片URL="",
        视频URL="",
        音频URL="",
        返回尾帧=False,
        联网增强=False,
        回调URL="",
    ):
        api_key = API密钥.strip()
        if not api_key:
            raise ValueError("缺少 API 密钥")
        images = [图片1, 图片2, 图片3, 图片4, 图片5, 图片6, 图片7, 图片8, 图片9]
        image_urls, temp_files = _prepare_uploaded_url_list(api_key, images, 图片URL)
        try:
            reference_video_urls = _split_url_list(视频URL)
            reference_audio_urls = _split_url_list(音频URL)
            model_id, input_payload = _build_seedance_payload(
                模型,
                提示词,
                画幅,
                分辨率,
                时长秒,
                image_urls,
                reference_video_urls,
                reference_audio_urls,
                模式,
                生成音频,
                返回尾帧,
                联网增强,
            )
            task_id, raw = _submit_task(api_key, model_id, input_payload, 回调URL)
            request_payload = {"model": model_id, "input": input_payload}
            _save_video_submit_record(task_id, 模型, request_payload, "jobs")
            return (task_id, "jobs", json.dumps(request_payload, ensure_ascii=False), json.dumps(raw, ensure_ascii=False))
        finally:
            for temp_file in temp_files:
                if temp_file and os.path.exists(temp_file):
                    os.remove(temp_file)


class KieVideoUnifiedAsyncSubmit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "API密钥": ("STRING", {"multiline": False, "default": ""}),
                "模型": (VIDEO_ASYNC_MODELS, {"default": "seedance-2"}),
                "提示词": ("STRING", {"multiline": True, "default": ""}),
                "画幅": (["16:9", "9:16", "1:1", "4:3", "3:4", "2:3", "3:2"], {"default": "16:9"}),
                "分辨率": (["480p", "720p", "1080p", "4K"], {"default": "720p"}),
                "时长秒": ("INT", {"default": 10, "min": 1, "max": 30, "step": 1}),
            },
            "optional": {
                "图片1": ("IMAGE",),
                "图片2": ("IMAGE",),
                "图片3": ("IMAGE",),
                "图片4": ("IMAGE",),
                "图片5": ("IMAGE",),
                "图片6": ("IMAGE",),
                "图片7": ("IMAGE",),
                "图片8": ("IMAGE",),
                "图片9": ("IMAGE",),
                "视频": ("VIDEO",),
                "音频": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("任务ID", "查询类型", "请求JSON", "原始JSON")
    FUNCTION = "submit"
    CATEGORY = "KieAI/视频"
    IS_CHANGED = _force_run_token

    def submit(
        self,
        API密钥,
        模型,
        提示词,
        画幅,
        分辨率,
        时长秒,
        图片1=None,
        图片2=None,
        图片3=None,
        图片4=None,
        图片5=None,
        图片6=None,
        图片7=None,
        图片8=None,
        图片9=None,
        视频=None,
        音频=None,
    ):
        api_key = API密钥.strip()
        if not api_key:
            raise ValueError("缺少 API 密钥")

        images = [图片1, 图片2, 图片3, 图片4, 图片5, 图片6, 图片7, 图片8, 图片9]
        image_urls, temp_files = _prepare_uploaded_url_list(api_key, images, "")
        video_url, video_temp_files = _upload_media_input(api_key, 视频, "video")
        audio_url, audio_temp_files = _upload_media_input(api_key, 音频, "audio")
        temp_files.extend(video_temp_files)
        temp_files.extend(audio_temp_files)
        video_urls = [video_url] if video_url else []
        audio_urls = [audio_url] if audio_url else []

        try:
            query_type = "jobs"
            if 模型 in VEO_MODEL_MAP:
                _ensure_no_items(模型, "视频", video_urls)
                _ensure_no_items(模型, "音频", audio_urls)
                if not image_urls:
                    _ensure_no_items(模型, "图片", image_urls)
                    ref_mode = False
                elif len(image_urls) <= 2:
                    _ensure_max_items(模型, "首尾帧图片", image_urls, 2)
                    ref_mode = False
                elif len(image_urls) <= 3:
                    _ensure_max_items(模型, "参考图片", image_urls, 3)
                    ref_mode = True
                else:
                    raise ValueError(f"{模型} 最多支持 3 张图片，当前传入 {len(image_urls)} 张。")
                input_payload = _build_veo_payload(模型, 提示词, 画幅, 分辨率, image_urls, 0, "", ref_mode)
                task_id, raw = _submit_veo_task(api_key, input_payload, "")
                request_payload = dict(input_payload)
                query_type = "veo"

            elif 模型.startswith("grok"):
                _ensure_no_items(模型, "视频", video_urls)
                _ensure_no_items(模型, "音频", audio_urls)
                model_id, input_payload = _build_grok_payload(
                    模型,
                    提示词,
                    画幅,
                    分辨率,
                    时长秒,
                    image_urls,
                    "",
                    0,
                    True,
                )
                task_id, raw = _submit_task(api_key, model_id, input_payload, "")
                request_payload = {"model": model_id, "input": input_payload}

            elif 模型 in SEEDANCE_MODEL_MAP:
                model_id, input_payload = _build_seedance_payload(
                    模型,
                    提示词,
                    画幅,
                    分辨率,
                    时长秒,
                    image_urls,
                    video_urls,
                    audio_urls,
                    "自动",
                    False,
                    False,
                    True,
                )
                task_id, raw = _submit_task(api_key, model_id, input_payload, "")
                request_payload = {"model": model_id, "input": input_payload}

            else:
                raise ValueError(f"不支持的模型: {模型}")

            _save_video_submit_record(task_id, 模型, request_payload, query_type)
            return (task_id, query_type, json.dumps(request_payload, ensure_ascii=False), json.dumps(raw, ensure_ascii=False))
        finally:
            for temp_file in temp_files:
                if temp_file and os.path.exists(temp_file):
                    os.remove(temp_file)


class KieVideoSeriesAsyncQuery:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "API密钥": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("任务报告",)
    FUNCTION = "query"
    CATEGORY = "KieAI/视频"
    IS_CHANGED = _force_run_token

    def query(self, API密钥):
        api_key = API密钥.strip()
        if not api_key:
            raise ValueError("缺少 API 密钥")
        tasks = _read_tasks()
        video_tasks = {
            task_id: record
            for task_id, record in tasks.items()
            if isinstance(record, dict) and record.get("query_type") in {"jobs", "veo"}
        }
        if not video_tasks:
            return ("暂无已提交的视频任务。",)

        refreshed = 0
        query_errors = 0
        for task_id, record in video_tasks.items():
            status = record.get("status", "submitted")
            if status in {"fail", "downloaded"}:
                continue
            if status == "success" and record.get("result_urls"):
                continue
            try:
                _query_saved_video_task(api_key, task_id, record)
                _save_task(task_id, {"last_query_error": ""})
                refreshed += 1
            except Exception as exc:
                _save_task(task_id, {"last_query_error": str(exc)})
                query_errors += 1

        tasks = _read_tasks()
        video_tasks = [
            (task_id, record)
            for task_id, record in tasks.items()
            if isinstance(record, dict) and record.get("query_type") in {"jobs", "veo"}
        ]
        video_tasks.sort(
            key=lambda item: (
                item[1].get("submit_order", 0),
                item[1].get("submitted_at", ""),
            ),
            reverse=True,
        )

        counts = {"running": 0, "ready": 0, "downloaded": 0, "failed": 0}
        lines = [
            f"全部视频任务: {len(video_tasks)} | 本次查询: {refreshed} | 查询错误: {query_errors}"
        ]
        for task_id, record in video_tasks:
            status = record.get("status", "submitted")
            error = record.get("error") or record.get("last_query_error") or ""
            if status == "downloaded":
                label = "已下载"
                counts["downloaded"] += 1
            elif status == "success" and record.get("result_urls"):
                label = "成功待下载"
                counts["ready"] += 1
            elif status == "fail":
                label = "失败"
                counts["failed"] += 1
            else:
                label = "生成中"
                counts["running"] += 1
            line = f"[{label}] {record.get('model', '未知模型')} | {task_id}"
            if error:
                line += f" | 原因: {error}"
            lines.append(line)

        lines.insert(
            1,
            "生成中: {running} | 成功待下载: {ready} | 已下载: {downloaded} | 失败: {failed}".format(
                **counts
            ),
        )
        return ("\n".join(lines),)


NODE_CLASS_MAPPINGS = {
    "KieRefreshModelPresets": KieRefreshModelPresets,
    "KieUniversalSubmit": KieUniversalSubmit,
    "KieUniversalQuery": KieUniversalQuery,
    "KieUniversalDownload": KieUniversalDownload,
    "KieVideoUnifiedAsyncSubmit": KieVideoUnifiedAsyncSubmit,
    "KieVideoSeriesAsyncQuery": KieVideoSeriesAsyncQuery,
    "KieVideoResultDownload": KieVideoResultDownload,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "KieRefreshModelPresets": "Kie 刷新模型列表（非文本）",
    "KieUniversalSubmit": "Kie 通用模型提交",
    "KieUniversalQuery": "Kie 通用任务查询",
    "KieUniversalDownload": "Kie 通用结果下载",
    "KieVideoUnifiedAsyncSubmit": "Kie 视频统一异步提交",
    "KieVideoSeriesAsyncQuery": "Kie 视频系列异步查询",
    "KieVideoResultDownload": "Kie 视频结果下载",
}
