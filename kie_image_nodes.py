import io
import json
import os
import random
import ssl
import tempfile
import time

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
KIE_FILE_HOST = "https://kieai.redpandaai.co"

MODEL_CHOICES = [
    "GPT Image-2",
    "Nano Banana",
    "Nano Banana 2",
    "Nano Banana Pro",
    "Seedream 5 Lite",
]

ASPECT_RATIO_CHOICES = [
    "auto",
    "1:1",
    "16:9",
    "9:16",
    "4:3",
    "3:4",
    "2:3",
    "3:2",
    "4:5",
    "5:4",
    "21:9",
    "1:4",
    "4:1",
    "1:8",
    "8:1",
]

RESOLUTION_CHOICES = ["自动", "1K", "2K", "4K"]

IMAGE_MODEL_RULES = {
    "GPT Image-2": {
        "max_images": 9,
        "text_model": "gpt-image-2-text-to-image",
        "image_model": "gpt-image-2-image-to-image",
        "image_field": "input_urls",
    },
    "Nano Banana": {
        "max_images": 9,
        "text_model": "google/nano-banana",
        "image_model": "google/nano-banana-edit",
        "image_field": "image_urls",
    },
    "Nano Banana 2": {
        "max_images": 9,
        "text_model": "nano-banana-2",
        "image_model": "nano-banana-2",
        "image_field": "image_input",
    },
    "Nano Banana Pro": {
        "max_images": 9,
        "text_model": "nano-banana-pro",
        "image_model": "nano-banana-pro",
        "image_field": "image_input",
    },
    "Seedream 5 Lite": {
        "max_images": 9,
        "text_model": "seedream/5-lite-text-to-image",
        "image_model": "seedream/5-lite-image-to-image",
        "image_field": "image_urls",
    },
}


def _get_headers(api_key):
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key.strip()}",
    }


def _split_url_list(value):
    if not value:
        return []
    if isinstance(value, list):
        items = value
    else:
        text = str(value).replace("\r", "\n").replace(",", "\n")
        items = text.split("\n")
    return [item.strip() for item in items if item and item.strip()]


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


def _upload_file_stream(api_key, file_path):
    if not file_path or not os.path.exists(file_path):
        return None

    upload_name = os.path.basename(file_path) or f"kie_{int(time.time())}_{random.randint(100, 999)}.png"
    with open(file_path, "rb") as file_obj:
        response = requests.post(
            f"{KIE_FILE_HOST}/api/file-stream-upload",
            headers={"Authorization": f"Bearer {api_key.strip()}"},
            data={"uploadPath": "comfy", "fileName": upload_name},
            files={"file": (upload_name, file_obj)},
            timeout=120,
            verify=False,
        )
    response.raise_for_status()
    data = response.json()
    if data.get("success"):
        return data.get("data", {}).get("downloadUrl")
    raise ValueError(data.get("message") or data.get("msg") or "图片上传失败")


def _prepare_image_urls(api_key, images, image_url_text):
    urls = []
    temp_files = []

    for image in images:
        if image is None:
            continue
        temp_file = _tensor_to_temp_file(image)
        temp_files.append(temp_file)
        urls.append(_upload_file_stream(api_key, temp_file))

    urls.extend(_split_url_list(image_url_text))

    deduped = []
    seen = set()
    for url in urls:
        if url and url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped, temp_files


def _normalize_aspect_ratio(value, family_name):
    aspect_ratio = (value or "auto").strip()
    if family_name == "Seedream 5 Lite" and aspect_ratio == "auto":
        return "1:1"
    return aspect_ratio


def _normalize_resolution(value):
    resolution = (value or "自动").strip()
    if resolution not in {"1K", "2K", "4K"}:
        return ""
    return resolution


def _seedream_quality(resolution):
    return "high" if resolution == "4K" else "basic"


def _validate_image_count(model_family, image_urls):
    rule = IMAGE_MODEL_RULES.get(model_family)
    if not rule:
        return
    count = len([url for url in image_urls if url])
    max_images = rule["max_images"]
    if count > max_images:
        raise ValueError(f"{model_family} 的参考图片最多支持 {max_images} 个，当前传入 {count} 个。请删除多余输入后重试。")


def _build_payload(model_family, prompt, aspect_ratio, resolution, image_urls):
    image_urls = [url for url in image_urls if url]
    _validate_image_count(model_family, image_urls)
    has_images = bool(image_urls)
    normalized_ratio = _normalize_aspect_ratio(aspect_ratio, model_family)
    normalized_resolution = _normalize_resolution(resolution)
    rule = IMAGE_MODEL_RULES.get(model_family)

    if model_family == "GPT Image-2":
        if has_images:
            payload = {
                "prompt": prompt,
                rule["image_field"]: image_urls,
                "aspect_ratio": normalized_ratio or "auto",
            }
            if normalized_resolution:
                payload["resolution"] = normalized_resolution
            return rule["image_model"], payload

        payload = {
            "prompt": prompt,
            "aspect_ratio": normalized_ratio or "auto",
        }
        if normalized_resolution:
            payload["resolution"] = normalized_resolution
        return rule["text_model"], payload

    if model_family == "Nano Banana":
        if has_images:
            return rule["image_model"], {
                "prompt": prompt,
                rule["image_field"]: image_urls,
                "output_format": "png",
                "aspect_ratio": normalized_ratio or "auto",
            }

        return rule["text_model"], {
            "prompt": prompt,
            "output_format": "png",
            "aspect_ratio": normalized_ratio or "auto",
        }

    if model_family == "Nano Banana 2":
        payload = {
            "prompt": prompt,
            rule["image_field"]: image_urls,
            "aspect_ratio": normalized_ratio or "auto",
            "output_format": "jpg",
        }
        if normalized_resolution:
            payload["resolution"] = normalized_resolution
        return rule["image_model"] if has_images else rule["text_model"], payload

    if model_family == "Nano Banana Pro":
        payload = {
            "prompt": prompt,
            rule["image_field"]: image_urls,
            "aspect_ratio": normalized_ratio or "auto",
            "output_format": "png",
        }
        if normalized_resolution:
            payload["resolution"] = normalized_resolution
        return rule["image_model"] if has_images else rule["text_model"], payload

    if model_family == "Seedream 5 Lite":
        common = {
            "prompt": prompt,
            "aspect_ratio": normalized_ratio or "1:1",
            "quality": _seedream_quality(normalized_resolution),
            "nsfw_checker": False,
        }
        if has_images:
            common[rule["image_field"]] = image_urls
            return rule["image_model"], common
        return rule["text_model"], common

    raise ValueError(f"不支持的模型: {model_family}")


def _submit_task(api_key, model_name, payload):
    response = requests.post(
        f"{KIE_API_HOST}/api/v1/jobs/createTask",
        headers=_get_headers(api_key),
        json={"model": model_name, "input": payload},
        timeout=60,
        verify=False,
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
    response = requests.get(
        f"{KIE_API_HOST}/api/v1/jobs/recordInfo",
        headers=_get_headers(api_key),
        params={"taskId": task_id},
        timeout=30,
        verify=False,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("code") != 200:
        raise ValueError(data.get("msg") or "查询任务失败")
    return data.get("data", {})


def _task_state(task_data):
    success_flag = task_data.get("successFlag")
    if success_flag == 1:
        return "success"
    if success_flag in {2, 3}:
        return "failed"

    state = str(task_data.get("state") or task_data.get("status") or "").strip().lower()
    if state in {"success", "succeeded", "completed", "done"}:
        return "success"
    if state in {"fail", "failed", "error", "canceled", "cancelled"}:
        return "failed"
    return "waiting"


def _parse_json_value(value):
    if isinstance(value, str):
        text = value.strip()
        if text and text[0] in "[{":
            try:
                return json.loads(text)
            except Exception:
                return value
    return value


def _collect_urls(value, results):
    parsed = _parse_json_value(value)
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


def _add_urls_from_value(value, results):
    if value is None:
        return
    if isinstance(value, (list, dict, str)):
        _collect_urls(value, results)


def _looks_like_image_url(url):
    lowered = url.lower()
    return any(
        token in lowered
        for token in [".png", ".jpg", ".jpeg", ".webp", "image", "img", "file.aiquickdraw.com"]
    )


def _extract_result_url(task_data):
    candidates = []

    response_data = _parse_json_value(task_data.get("response")) or {}
    info_data = _parse_json_value(task_data.get("info")) or {}
    result_json = _parse_json_value(task_data.get("resultJson"))

    for value in [
        task_data.get("resultUrls"),
        task_data.get("imageUrls"),
        task_data.get("images"),
        response_data.get("resultUrls") if isinstance(response_data, dict) else None,
        response_data.get("imageUrls") if isinstance(response_data, dict) else None,
        response_data.get("images") if isinstance(response_data, dict) else None,
        info_data.get("resultUrls") if isinstance(info_data, dict) else None,
        info_data.get("imageUrls") if isinstance(info_data, dict) else None,
        result_json,
        task_data,
    ]:
        _add_urls_from_value(value, candidates)

    deduped = []
    seen = set()
    for url in candidates:
        if url and url not in seen:
            seen.add(url)
            deduped.append(url)

    for url in deduped:
        if _looks_like_image_url(url):
            return url
    return deduped[0] if deduped else ""


def _task_error(task_data):
    return task_data.get("failMsg") or task_data.get("errorMessage") or task_data.get("message") or ""


def _download_image_tensor(url):
    response = requests.get(url, timeout=180, verify=False)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content))
    if image.mode != "RGB":
        image = image.convert("RGB")
    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array)[None,]


class KieImageNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "API密钥": ("STRING", {"multiline": False, "default": ""}),
                "模型": (MODEL_CHOICES, {"default": "GPT Image-2"}),
                "提示词": ("STRING", {"multiline": True, "default": "A beautiful image."}),
                "画面比例": (ASPECT_RATIO_CHOICES, {"default": "auto"}),
                "分辨率": (RESOLUTION_CHOICES, {"default": "自动"}),
            },
            "optional": {
                "图片": ("IMAGE",),
                "图片2": ("IMAGE",),
                "图片3": ("IMAGE",),
                "图片4": ("IMAGE",),
                "图片5": ("IMAGE",),
                "图片6": ("IMAGE",),
                "图片7": ("IMAGE",),
                "图片8": ("IMAGE",),
                "图片9": ("IMAGE",),
                "图片URL": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("图像", "结果链接", "原始JSON")
    FUNCTION = "generate"
    CATEGORY = "KieAI/图像"

    def generate(self, **kwargs):
        api_key = (kwargs.get("API密钥") or "").strip()
        model_family = kwargs.get("模型") or MODEL_CHOICES[0]
        prompt = (kwargs.get("提示词") or "").strip()
        aspect_ratio = kwargs.get("画面比例") or "auto"
        resolution = kwargs.get("分辨率") or "自动"
        images = [
            kwargs.get("图片"),
            kwargs.get("图片2"),
            kwargs.get("图片3"),
            kwargs.get("图片4"),
            kwargs.get("图片5"),
            kwargs.get("图片6"),
            kwargs.get("图片7"),
            kwargs.get("图片8"),
            kwargs.get("图片9"),
        ]
        image_url_text = kwargs.get("图片URL") or ""

        if not api_key:
            raise ValueError("缺少 API 密钥")
        if not prompt:
            raise ValueError("提示词不能为空")

        image_urls, temp_files = _prepare_image_urls(api_key, images, image_url_text)
        try:
            model_name, input_payload = _build_payload(
                model_family,
                prompt,
                aspect_ratio,
                resolution,
                image_urls,
            )
            task_id, _submit_data = _submit_task(api_key, model_name, input_payload)

            for _ in range(180):
                time.sleep(4)
                task_data = _query_task(api_key, task_id)
                state = _task_state(task_data)
                if state == "success":
                    image_url = _extract_result_url(task_data)
                    if not image_url:
                        raise ValueError("任务已完成，但没有取到图片链接")
                    image_tensor = _download_image_tensor(image_url)
                    return image_tensor, image_url, json.dumps(task_data, ensure_ascii=False)
                if state == "failed":
                    raise ValueError(_task_error(task_data) or "图像生成失败")

            raise TimeoutError(f"等待超时，任务仍在处理中: {task_id}")
        finally:
            for temp_file in temp_files:
                if temp_file and os.path.exists(temp_file):
                    os.remove(temp_file)


NODE_CLASS_MAPPINGS = {
    "KieImageNode": KieImageNode,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "KieImageNode": "Kie 图像生成",
}
