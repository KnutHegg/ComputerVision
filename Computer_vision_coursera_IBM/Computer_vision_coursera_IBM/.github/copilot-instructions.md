<!-- Copilot / AI agent instructions for this workspace -->
# Repo snapshot

- Small learning repo containing one notebook: [Module_2/Image_processing_with_Pillow.ipynb](Module_2/Image_processing_with_Pillow.ipynb).
- Image assets live alongside the notebook (lenna.png, baboon.png, barbara.png).

# Purpose for AI agents

This repository is a single lab notebook demonstrating basic image processing using Pillow (PIL). The goal of an AI coding agent here is to help edit, refactor, or extract runnable Python modules from the notebook, and to produce concise guidance and runnable commands for users to reproduce examples locally.

# Key patterns and conventions (do not invent other structure)

- Canonical entry point is the notebook at [Module_2/Image_processing_with_Pillow.ipynb](Module_2/Image_processing_with_Pillow.ipynb).
- Top-level helper functions are defined inline in notebook cells (for example `get_concat_h()` in an early cell). When extracting code to scripts, preserve those helpers near the top of the new module.
- Image files are referenced by filename (e.g., `lenna.png`) and expected to live in the same directory as the notebook.
- Notebook uses shell magic for environment setup and downloads (e.g. `!pip install Pillow`, `!wget ...`). Prefer to convert these into standard `pip` commands when producing instructions or scripts.

# Typical developer workflows (commands agents should suggest)

- Create a venv and install dependencies (recommended):

    python -m venv .venv
    .venv\Scripts\activate
    pip install Pillow matplotlib numpy

- To run the examples interactively: open the notebook in Jupyter/Lab and run the cells; the notebook already downloads example images with `wget` in cells but the repo contains the images.
- To convert cells into a runnable script: extract cells in order, keep imports at top (`from PIL import Image, ImageOps`, `import numpy as np`, `import matplotlib.pyplot as plt`), and replace notebook magics (`!pip`, `!wget`) with equivalent shell or Python code.

# Important code examples (from the notebook)

- Helper concat: `def get_concat_h(im1, im2):` — used to display side-by-side results (keep when extracting display utilities).
- Loading images: `image = Image.open('lenna.png')` and `baboon = Image.open('baboon.png')`.
- Converting to grayscale: `image_gray = ImageOps.grayscale(image)`.
- Converting PIL → NumPy: `array = np.asarray(image)` and `array = np.array(image)` (note behavior difference; `np.array` makes a copy).

# Integration and external dependencies

- Main runtime libs used: `Pillow`, `numpy`, `matplotlib` (these are only dependencies discoverable in the notebook).
- The notebook uses `wget` to fetch images in CI-style cells — if reproducing outside notebook, download the images or use the local files already present in `Module_2/`.

# What to avoid / assumptions

- There is no package layout, tests, or CI in this repo. Do not attempt to add or assume an existing `setup.py`, `pyproject.toml`, or test framework unless the user asks.
- Do not change or remove the provided image assets; assume notebook demos expect them in the same folder.

# How to propose changes (agent behavior)

- For small fixes: propose a patch that edits the notebook or adds a single script `scripts/run_lab.py` that reproduces the examples. Keep changes minimal and documented in the PR description.
- For extraction: create a new folder `src/` or `scripts/` and add a single file that runs the notebook examples end-to-end. Include a short `README` snippet of commands to run and the `pip` install lines above.

If any section is unclear or you want additional examples (for example: a ready-to-run `scripts/run_lab.py`), tell me which format you prefer and I will draft it.
