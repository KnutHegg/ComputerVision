# Copilot instructions — Computer_Vision_Exercises

These instructions help an AI coding agent be productive in this repository. Keep suggestions concrete, minimal, and reproducible.

**Project Overview**
- **Repo layout**: experiments and exercises are organized by chapter under [Chapter_3](Chapter_3/). Primary work is in Jupyter notebooks (e.g., [Chapter_3/Color_Balance.ipynb](Chapter_3/Color_Balance.ipynb)).
- **Assets**: sample images live under [Files/Random_Images](Files/Random_Images/) (example: `houses_1.avif`). Treat these as read-only test inputs unless a task explicitly requires adding new images.

**What to change and how**
- **Edit notebooks, not plain .py scripts**: Most exercises are Jupyter notebooks. When adding or modifying cells, preserve existing cell metadata (each existing cell has an `id` field). New cells should include `metadata.language: "python"` and follow the notebook JSON conventions used in this repo.
- **Small, focused changes**: Make minimal edits to a notebook cell, run the notebook locally to validate, and commit only the changed notebook. Avoid reordering or regenerating every cell ID.

**Patterns & conventions discovered**
- **Notebook-first workflow**: Solutions and examples are implemented inside `.ipynb` files. Provide code snippets as self-contained notebook cells (include imports and small runnable examples).
- **Preserve notebook structure**: Do not remove top-level metadata or rewrite the whole notebook. Use diffs that change only the necessary cells.
- **Image assets**: The repo uses modern image formats (AVIF). Use libraries that support AVIF (e.g., Pillow with avif plugin or OpenCV builds that support it), or convert to a common format only when explicitly requested.

**Examples to reference**
- To update color-balance logic, edit [Chapter_3/Color_Balance.ipynb](Chapter_3/Color_Balance.ipynb) and add a new code cell demonstrating the function usage on [Files/Random_Images/houses_1.avif](Files/Random_Images/houses_1.avif).

**Testing & validation**
- Validate changes by running the edited notebook in VS Code or Jupyter. There are no repository-level test scripts; verify outputs visually and by running the modified cells.

**Commit and PR guidance**
- Keep commits focused: one notebook feature or fix per commit/PR.
- In the PR description, reference the notebook and any images used for verification.

If anything here is unclear or if you want me to include environment/setup commands (Python version, recommended packages, or conda/venv instructions), tell me which environment you use and I will add them.
