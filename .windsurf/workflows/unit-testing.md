---
description: Automatic Unit Testing Workflow
---

Automated Unit Test Generation Workflow

Detect New or Modified Function

Whenever a new function or class is added to any module under src/pipeline/, Windsurf must recognize the change based on Git diff or file modification timestamps.

Explicitly identify the function or class name, its signature (parameters and return type), and a brief description from its docstring.

Determine Test Module and File Name

For each module, map module_name.py ➔ tests/test_module_name.py.

If tests/test_module_name.py does not exist, create it with an initial header and import statements:

python
Copy
Edit
import pytest
from pipeline.module_name import function_name
If it already exists, append new test functions without overwriting existing tests.

Extract Input/Output Specification

Read the function’s docstring (Google-style or NumPy-style) to extract:

Parameter names and expected types.

Brief description of behavior.

Any “Raises” or “Returns” sections.

If no docstring is provided, use type hints or default values to infer basic input and output types.

Create Normal Case Test

Generate a test function named test_<module>_<function>_normal(). For example, for calculate_dbh(points), create:

python
Copy
Edit
def test_calculate_dbh_normal():
    # Arrange: create a small sample cloud with known DBH
    points = ...  # construct minimal PointCloud or NumPy array
    # Act
    result = calculate_dbh(points)
    # Assert: numeric tolerance for expected DBH
    assert pytest.approx(expected_value, rel=1e-3) == result
Use simple synthetic data for which the output is well known:

For numeric functions, small arrays or handcrafted values.

For clustering, a minimal set of 3–4 points forming one cluster.

Insert a comment in English followed by Spanish. For example:

python
Copy
Edit
# Ensure DBH is correctly calculated on a perfect cylinder  # Asegura que DBH se calcule correctamente en un cilindro perfecto
Create Edge Case Test

Generate a test named test_<module>_<function>_edge(). Identify potential edge conditions based on docstring or function signature:

Empty inputs (empty list or DataFrame).

Minimal valid input (e.g., only one point).

Extremely large or small values.

Example for load_las(path):

python
Copy
Edit
def test_load_las_nonexistent_file(tmp_path):
    # Arrange: define a path that does not exist  # Definir ruta que no exista
    fake_path = tmp_path / "does_not_exist.las"
    # Act & Assert: expect FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_las(str(fake_path))
Generate Fixtures (If Needed)

If a function depends on a common setup (e.g., a small sample LAS/LAZ file), Windsurf should:

Create a fixture in tests/conftest.py or inline at the top of the test file.

For example:

python
Copy
Edit
@pytest.fixture
def sample_pointcloud(tmp_path):
    # Create a minimal LAS with laspy  # Crear un LAS mínimo con laspy
    file = tmp_path / "sample.las"
    # ... code to write 5 points with known coordinates ...
    return str(file)
Parameterize Multiple Scenarios

Where appropriate, use @pytest.mark.parametrize to test the function with several inputs.

Template:

python
Copy
Edit
@pytest.mark.parametrize(
    "input_data, expected_output",
    [
        (data_case1, expected1),  # Simplest valid case  # Caso válido más simple
        (data_case2, expected2),  # More complex scenario  # Escenario más complejo
    ],
)
def test_function_various(input_data, expected_output):
    result = function_under_test(input_data)
    assert result == expected_output
Insert Spanish and English Comments

For every generated test block, precede each logical step with two comment lines:

English description (e.g., # Arrange minimal point set for sweep calculation)

Spanish equivalent (e.g., # Preparar conjunto mínimo de puntos para cálculo de sweep)

Keep comments concise, no longer than 80 characters per line.

Write Assertions According to Return Types

Numeric outputs: use pytest.approx(expected, rel=1e-3).

Boolean or categorical outputs: use assert result == expected_category.

Collections (lists, sets): use assert set(result) == set(expected_list).

Pandas DataFrame: assert_frame_equal(result_df, expected_df) with pandas.testing.

Run and Validate Tests Locally

After generating each test file, Windsurf triggers a local pytest run (e.g., pytest tests/test_module_name.py -q).

Capture any failures; if a test fails, annotate the failure with comments explaining likely causes and suggestions to fix.

Commit New Tests Automatically

Once tests pass, stage and commit the new test file with a descriptive message:

sql
Copy
Edit
git add tests/test_module_name.py
git commit -m "Add unit tests for <module>.py: <function_name>"
CI Integration

Ensure that generated tests are included in the GitHub Actions pipeline.

If a new test fails on CI, post a concise log comment prompting the developer to investigate:

cpp
Copy
Edit
[CI][ERROR] test_module_name::test_function_edge FAILED
Check edge-case handling for <function_name> in module_name.
Repeat for Subsequent Modifications

On every pull request or code change, rerun Steps 1–12 to update or add new tests.

If a function signature changes, Windsurf must modify existing tests accordingly (parameter names, expected behavior).

Document Test Coverage Goals

Maintain a coverage/ badge or report:

text
Copy
Edit
Required coverage: ≥ 85% for lines in src/pipeline/
Generate a coverage report as part of the CI step:

bash
Copy
Edit
pytest --cov=src/pipeline --cov-report=xml
Developer Notification

After tests generation and CI pass, notify via a summary comment in the PR:

csharp
Copy
Edit
[AI Assist] Unit tests for module_name.py have been added and passed. Coverage: 90%.