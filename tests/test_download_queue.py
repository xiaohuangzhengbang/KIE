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
                video, _ = MODULE.KieVideoResultDownload().download()
                saved = json.loads(task_file.read_text(encoding="utf-8"))

                self.assertTrue(all(task["status"] == "downloaded" for task in saved.values()))
                self.assertEqual(len(list(root.glob("kie_*.mp4"))), 10)
                self.assertTrue(video.video_path.endswith("task10.mp4"))
                with self.assertRaises(ValueError):
                    MODULE.KieVideoResultDownload().download()
                self.assertEqual(len(list(root.glob("kie_*.mp4"))), 10)


if __name__ == "__main__":
    unittest.main()
