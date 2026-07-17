import numpy as np
import pytest

from pt_tem_cv.sam_prompting import detector_boxes_to_sam_prompts


def test_source_boxes_become_prompt_array() -> None:
    prompts = detector_boxes_to_sam_prompts([[2, 3, 12, 15]], image_width=20, image_height=20)
    assert prompts.dtype == np.float32
    assert prompts.tolist() == [[2.0, 3.0, 12.0, 15.0]]


def test_out_of_bounds_prompt_is_rejected() -> None:
    with pytest.raises(ValueError, match="within"):
        detector_boxes_to_sam_prompts([[2, 3, 21, 15]], image_width=20, image_height=20)
