# Human Verification in the Chapter 5 Workflow

Automation proposed particle detections/masks, substrate masks, and image-scale readings. Human involvement was retained at the points where the scientific interpretation depends on image-specific context: scale-bar value/unit checks and quality review of the resulting records. Physical-unit conversion requires an `accepted` or `verified` scale record.

This verification is represented explicitly in `configs/ch5_ml_tem/scale_verification_schema.yaml` and enforced by `pt_tem_cv.scale.calibrated_nm_per_pixel`. The repository does not describe human checking as additional model training or as an independent experimental measurement.
