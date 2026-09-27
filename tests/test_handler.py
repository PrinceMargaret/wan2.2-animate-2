import base64
import tempfile
import unittest
from pathlib import Path

import handler as handler_module
from handler import handler


API_WORKFLOW = {
    "6": {
        "inputs": {"text": "a ball", "clip": ["30", 1]},
        "class_type": "CLIPTextEncode",
    }
}


class HandlerTests(unittest.TestCase):
    def test_dry_run_returns_the_submitted_workflow(self):
        result = handler({"id": "job", "input": {"dry_run": True, "workflow": API_WORKFLOW}})
        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(result["node_count"], 1)
        self.assertEqual(result["workflow"]["6"]["class_type"], "CLIPTextEncode")

    def test_accepts_workflow_json_string(self):
        result = handler({"id": "job", "input": {"dry_run": True, "workflow": '{"9": {"inputs": {"image": "ref.png"}, "class_type": "LoadImage"}}'}})
        self.assertEqual(result["workflow"]["9"]["inputs"]["image"], "ref.png")

    def test_unwraps_prompt_wrapper(self):
        result = handler({"id": "job", "input": {"dry_run": True, "workflow": {"prompt": API_WORKFLOW}}})
        self.assertIn("6", result["workflow"])

    def test_rejects_editor_workflow(self):
        result = handler({"id": "job", "input": {"workflow": {"nodes": [], "links": []}}})
        self.assertIn("Export (API)", result["error"])

    def test_missing_workflow_is_an_error(self):
        result = handler({"id": "job", "input": {}})
        self.assertIn("workflow is required", result["error"])

    def test_inline_image_in_workflow_is_saved_as_a_filename(self):
        payload = base64.b64encode(b"png-bytes").decode()
        original_dir = handler_module.COMFY_INPUT_DIR
        with tempfile.TemporaryDirectory() as tmp:
            handler_module.COMFY_INPUT_DIR = Path(tmp)
            result = handler(
                {
                    "id": "job",
                    "input": {
                        "dry_run": True,
                        "workflow": {
                            "189": {
                                "inputs": {"image": f"data:image/png;base64,{payload}"},
                                "class_type": "LoadImage",
                            },
                            "6": {
                                "inputs": {"text": "keep this prompt", "clip": ["30", 1]},
                                "class_type": "CLIPTextEncode",
                            },
                        },
                    },
                }
            )
        handler_module.COMFY_INPUT_DIR = original_dir
        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(result["workflow"]["189"]["inputs"]["image"], "189_image_0.png")
        self.assertEqual(result["workflow"]["6"]["inputs"]["text"], "keep this prompt")
        self.assertEqual(result["workflow"]["6"]["inputs"]["clip"], ["30", 1])


if __name__ == "__main__":
    unittest.main()
