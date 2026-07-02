import importlib.util
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
