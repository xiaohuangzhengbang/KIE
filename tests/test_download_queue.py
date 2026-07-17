import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "kie_universal_nodes.py"
SPEC = importlib.util.spec_from_file_location("kie_universal_nodes_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DownloadQueueTests(unittest.TestCase):
    def test_uses_video_url_when_grok_returns_image_then_video(self):
        task = {
            "result_urls": [
                "https://example.test/input.png",
                "https://example.test/generated_video.mp4",
            ]
        }

        self.assertEqual(
            MODULE._video_result_url(task),
            "https://example.test/generated_video.mp4",
        )

    def test_selects_newest_successful_task(self):
        tasks = {
            "old": {
                "query_type": "jobs",
                "status": "success",
                "result_urls": ["https://example.test/old.mp4"],
                "submit_order": 1,
            },
            "new": {
                "query_type": "jobs",
                "status": "success",
                "result_urls": ["https://example.test/new.mp4"],
                "submit_order": 2,
            },
        }

        task_id, _ = MODULE._select_video_download_candidate(tasks)

        self.assertEqual(task_id, "new")

    def test_skips_url_that_was_already_downloaded(self):
        tasks = {
            "downloaded": {
                "query_type": "jobs",
                "status": "downloaded",
                "result_urls": ["https://example.test/video.mp4"],
                "video_path": "video.mp4",
                "submit_order": 1,
            },
            "duplicate": {
                "query_type": "jobs",
                "status": "success",
                "result_urls": ["https://example.test/video.mp4"],
                "submit_order": 2,
            },
        }

        self.assertIsNone(MODULE._select_video_download_candidate(tasks))

    def test_one_execution_downloads_all_ready_tasks_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_file = root / "tasks.json"
            tasks = {
                f"task{index}": {
                    "query_type": "jobs",
                    "status": "success",
                    "result_urls": [f"https://example.test/{index}.mp4"],
                    "submit_order": index,
                    "model": "test",
                }
                for index in range(1, 11)
            }
            task_file.write_text(json.dumps(tasks), encoding="utf-8")

            folder_paths = types.SimpleNamespace(get_output_directory=lambda: temp_dir)

            def fake_download(url, output_path):
                Path(output_path).write_bytes(url.encode("utf-8"))
                return True

            with (
                mock.patch.object(MODULE, "TASK_FILE", str(task_file)),
                mock.patch.object(MODULE, "_download", fake_download),
                mock.patch.dict(sys.modules, {"folder_paths": folder_paths}),
            ):
                videos, _ = MODULE.KieVideoResultDownload().download("全部待下载")
                saved = json.loads(task_file.read_text(encoding="utf-8"))

                self.assertTrue(all(task["status"] == "downloaded" for task in saved.values()))
                self.assertEqual(len(list(root.glob("kie_*.mp4"))), 10)
                self.assertEqual(len(videos), 10)
                self.assertTrue(videos[0].video_path.endswith("task10.mp4"))
                with self.assertRaises(ValueError):
                    MODULE.KieVideoResultDownload().download("全部待下载")
                self.assertEqual(len(list(root.glob("kie_*.mp4"))), 10)

    def test_single_mode_downloads_only_the_newest_task(self):
        tasks = {
            f"task{index}": {
                "query_type": "jobs",
                "status": "success",
                "result_urls": [f"https://example.test/{index}.mp4"],
                "submit_order": index,
                "model": "test",
            }
            for index in range(1, 4)
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_file = root / "tasks.json"
            task_file.write_text(json.dumps(tasks), encoding="utf-8")
            folder_paths = types.SimpleNamespace(get_output_directory=lambda: temp_dir)

            def fake_download(url, output_path):
                Path(output_path).write_bytes(url.encode("utf-8"))
                return True

            with (
                mock.patch.object(MODULE, "TASK_FILE", str(task_file)),
                mock.patch.object(MODULE, "_download", fake_download),
                mock.patch.dict(sys.modules, {"folder_paths": folder_paths}),
            ):
                videos, _ = MODULE.KieVideoResultDownload().download("单个最新")
                saved = json.loads(task_file.read_text(encoding="utf-8"))

                self.assertEqual(sum(task["status"] == "downloaded" for task in saved.values()), 1)
                self.assertEqual(len(videos), 1)
                self.assertTrue(videos[0].video_path.endswith("task3.mp4"))

    def test_batch_video_output_is_declared_as_a_list(self):
        self.assertEqual(MODULE.KieVideoResultDownload.OUTPUT_IS_LIST, (True, False))
        self.assertEqual(
            MODULE.KieUniversalDownload.OUTPUT_IS_LIST,
            (False, True, True, False),
        )

    def test_grok_imagine_allows_nine_reference_images(self):
        image_urls = [f"https://example.test/{index}.png" for index in range(9)]

        model_id, payload = MODULE._build_grok_payload(
            "grok-imagine",
            "prompt",
            "16:9",
            "720p",
            6,
            image_urls,
            "",
            0,
            True,
        )

        self.assertEqual(model_id, "grok-imagine/image-to-video")
        self.assertEqual(payload["image_urls"], image_urls)

    def test_grok_imagine_rejects_more_than_nine_reference_images(self):
        image_urls = [f"https://example.test/{index}.png" for index in range(10)]

        with self.assertRaises(ValueError) as error:
            MODULE._build_grok_payload(
                "grok-imagine",
                "prompt",
                "16:9",
                "720p",
                6,
                image_urls,
                "",
                0,
                True,
            )

        self.assertIn("9", str(error.exception))
        self.assertIn("10", str(error.exception))

    def test_grok_preview_still_rejects_multiple_images(self):
        image_urls = [f"https://example.test/{index}.png" for index in range(2)]

        with self.assertRaises(ValueError) as error:
            MODULE._build_grok_payload(
                "grok-imagine-video-1.5-preview",
                "prompt",
                "16:9",
                "720p",
                6,
                image_urls,
                "",
                0,
                True,
            )

        self.assertIn("1", str(error.exception))
        self.assertIn("2", str(error.exception))


if __name__ == "__main__":
    unittest.main()
