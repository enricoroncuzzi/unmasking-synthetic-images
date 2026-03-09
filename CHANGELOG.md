# Changelog

## [0.1.0] - 2026-03-09

### Added
- `data/generate_dataset.py`: forensic img2img laundering pipeline
- `requirements.txt`: minimal direct dependencies
- Dataset published on HuggingFace: `enricoroncuzzi/unmasking-synthetic-images-dataset`
  - 6000 images: 1000 real + 5000 laundered across SD1.5, SD2.1, SDXL Base, SD3.5, FLUX
  - `file_mappings.json`: maps each real image to its 5 laundered counterparts