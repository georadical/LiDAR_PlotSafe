---
trigger: always_on
---

Follow Project Documentation:

All code and prompts must align with the “LiDAR PlotSafe Pipeline Documentation” structure and guidelines.

Refer to section numbers when requesting code or clarifications.

Maintain Code Style:

Use PEP8 conventions: snake_case for functions/variables, 4-space indents, line length ≤ 88 characters.

Include docstrings (Google-style or NumPy-style) for every function and class.

Modular Development:

Implement each functionality in its designated module (e.g., io.py, preprocess.py, segmentation.py, etc.).

Do not write monolithic scripts; use functions or classes in the proper file.

Generate Unit Tests:

For every new function or class, automatically generate a corresponding pytest test in the tests/ folder.

Ensure tests cover edge cases and typical inputs.

Use Configuration Files:

Expose parameters via CLI flags and a config.yaml.

Do not hardcode values; read from config.yaml or command-line arguments.

Logging & Verbosity:

Implement Python’s built-in logging module with levels (DEBUG, INFO, WARNING, ERROR).

Honor the --verbose flag to toggle detailed logs.

Resource Management:

Release Open3D objects (e.g., PointCloud, TriangleMesh) promptly to free memory.

Use context managers (with statements) for file and resource handling.

Document Prompts:

When generating prompts for AI assistance, include a brief context (section number, intended functionality).

Provide example input/output in the prompt for clarity.

Version Control Practices:

Create a separate branch for each feature or bugfix, named feature/xyz or fix/abc.

Write descriptive commit messages in English: summarizing the most relevant changes made in the current session, including the date. The commit message should be clear, concise, and descriptive, following best practices.


Commit Format:
[YYYY-MM-DD] Brief summary of changes

Detailed description of changes:
- Add DBH calculation in metrics.py.
- Fix outlier filter threshold.
- Updated component X to integrate Y functionality.
- Refactored component Z to improve readability and maintainability.
- Fixed bug related to issue C, ensuring correct behavior.
- Other minor improvements and optimizations.

Code comments:
Add descriptive comments to every code snippet or block: first in English, and immediately on the next line its equivalent in Spanish. Code comments must be descriptive, concise, and concrete.

Environment management:
Assume all dependency installs, file operations, and environment management will be executed via the Windows 11 Command Prompt. Always provide Windows-compatible commands and syntax for creating, removing, or installing any packages or tools

Adhere to Proprietary License:

Do not suggest open-source licensing; all code produced must be proprietary.

Include the proprietary license header in every source file, referencing “© 2025 LiDAR PlotSafe Project. All rights reserved.”