import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "kie_image_nodes.py"
SPEC = importlib.util.spec_from_file_location("kie_image_nodes_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ImageRuleTests(unittest.TestCase):
    def test_image_node_exposes_nine_image_inputs(self):
        optional = MODULE.KieImageNode.INPUT_TYPES()["optional"]

        for index in range(1, 10):
            name = "图片" if index == 1 else f"图片{index}"
            self.assertIn(name, optional)

    def test_gpt_image_uses_input_urls_and_allows_nine_images(self):
        image_urls = [f"https://example.test/{index}.png" for index in range(9)]

        model, payload = MODULE._build_payload(
            "GPT Image-2",
            "prompt",
            "auto",
            "自动",
            image_urls,
        )

        self.assertEqual(model, "gpt-image-2-image-to-image")
        self.assertEqual(payload["input_urls"], image_urls)

    def test_nano_banana_uses_aspect_ratio_and_image_urls(self):
        model, payload = MODULE._build_payload(
            "Nano Banana",
            "prompt",
            "1:1",
            "自动",
            ["https://example.test/input.png"],
        )

        self.assertEqual(model, "google/nano-banana-edit")
        self.assertIn("aspect_ratio", payload)
        self.assertNotIn("image_size", payload)
        self.assertEqual(payload["image_urls"], ["https://example.test/input.png"])

    def test_nano_banana_pro_uses_image_input(self):
        model, payload = MODULE._build_payload(
            "Nano Banana Pro",
            "prompt",
            "1:1",
            "1K",
            ["https://example.test/input.png"],
        )

        self.assertEqual(model, "nano-banana-pro")
        self.assertEqual(payload["image_input"], ["https://example.test/input.png"])

    def test_seedream_switches_between_text_and_image_models(self):
        text_model, text_payload = MODULE._build_payload(
            "Seedream 5 Lite",
            "prompt",
            "1:1",
            "自动",
            [],
        )
        image_model, image_payload = MODULE._build_payload(
            "Seedream 5 Lite",
            "prompt",
            "1:1",
            "自动",
            ["https://example.test/input.png"],
        )

        self.assertEqual(text_model, "seedream/5-lite-text-to-image")
        self.assertNotIn("image_urls", text_payload)
        self.assertEqual(image_model, "seedream/5-lite-image-to-image")
        self.assertEqual(image_payload["image_urls"], ["https://example.test/input.png"])

    def test_image_models_reject_more_than_nine_images(self):
        image_urls = [f"https://example.test/{index}.png" for index in range(10)]

        with self.assertRaises(ValueError) as error:
            MODULE._build_payload("Nano Banana 2", "prompt", "auto", "1K", image_urls)

        self.assertIn("9", str(error.exception))
        self.assertIn("10", str(error.exception))


if __name__ == "__main__":
    unittest.main()
