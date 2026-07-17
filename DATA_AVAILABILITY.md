# Data Availability and Public-Release Boundary

This repository is a method-support package, not the complete research archive behind the dissertation.

## Included

- Chapter 5 source code for detector-side dataset construction, controlled augmentation, training, and evaluation.
- Public configuration and manifest templates with no dissertation image identifiers or frozen split membership.
- Selected Chapter 4 plotting scripts and compact provenance tables containing values reported in the dissertation.
- Data-free tests that create temporary synthetic fixtures only while the tests run.

## Not included

- Original TEM or STEM images, scale-bar crops, or literature image panels.
- Editable LabelMe annotations, support masks, or the identities of images assigned to training, validation, or evaluation subsets.
- Model weights, checkpoints, prediction files, per-particle tables, or production masks.
- Electrochemical raw data or instrument exports.
- Complete DFT coordinate and calculation archives or cluster logs.
- Dissertation Word/PDF files, review records, publisher PDFs, or licensed third-party figures.

These exclusions keep the public package aligned with the dissertation's stated lightweight scope and avoid redistributing files that are unnecessary for inspecting the released methods or that may be subject to size, copyright, or project-governance constraints. This repository does not make a separate promise that excluded materials will be publicly released.
