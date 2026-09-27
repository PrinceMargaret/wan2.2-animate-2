import unittest

import workflow
from handler import handler


def _options(**overrides):
    base = {
        "reference_image_name": "reference.png",
        "pose_video_name": "pose.mp4",
    }
    base.update(overrides)
    return base


class WorkflowTests(unittest.TestCase):
    def test_aligns_template_resolution_to_multiple_of_16(self):
        prompt = workflow.build_prompt(_options(width=482, height=854))
        resize = prompt["13"]["inputs"]["resize_type"]
        self.assertEqual(resize["width"], 480)
        self.assertEqual(resize["height"], 848)
        self.assertEqual(prompt["100"]["inputs"]["length"], 81)
        self.assertEqual(prompt["100"]["class_type"], "WanAnimate2ToVideo")
        self.assertNotIn("continue_motion", prompt["100"]["inputs"])
        self.assertEqual(prompt["911"]["class_type"], "SaveVideo")
        self.assertEqual(prompt["1"]["inputs"]["unet_name"], workflow.UNET_NAME)

    def test_second_chunk_continues_motion(self):
        prompt = workflow.build_prompt(_options(chunks=2, trim_duplicated_frame=False))
        self.assertEqual(prompt["120"]["inputs"]["continue_motion"], ["106", 0])
        self.assertEqual(prompt["120"]["inputs"]["video_frame_offset"], ["100", 5])
        self.assertNotIn("107", prompt)
        self.assertEqual(prompt["127"]["class_type"], "ImageFromBatch")
        self.assertEqual(prompt["900"]["inputs"]["images.image0"], ["106", 0])
        self.assertEqual(prompt["900"]["inputs"]["images.image1"], ["127", 0])

    def test_chunk_count_rejects_over_long_video(self):
        with self.assertRaises(ValueError):
            workflow.chunk_count(81 * 5, 81, True, max_chunks=4)

    def test_length_snaps_to_4n_plus_1(self):
        self.assertEqual(workflow.align_length(80), 77)
        self.assertEqual(workflow.align_length(81), 81)

    def test_dry_run_handler(self):
        result = handler({"id": "job", "input": {"dry_run": True, "match_video_length": True, "frame_count": 162, "max_chunks": 4}})
        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(result["chunks"], 2)
        self.assertIn("100", result["workflow"])
        self.assertIn("120", result["workflow"])

    def test_missing_media_is_an_error(self):
        result = handler({"id": "job", "input": {}})
        self.assertIn("reference_image", result["error"])


if __name__ == "__main__":
    unittest.main()
