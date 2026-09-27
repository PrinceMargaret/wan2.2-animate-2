import unittest

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


if __name__ == "__main__":
    unittest.main()
