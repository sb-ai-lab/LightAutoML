# Table of contents

- [Table of contents](#table-of-contents)
  - [Contributing to LightAutoML](#contributing-to-lightautoml)
    - [Fork and base branch](#fork-and-base-branch)
    - [Branch naming](#branch-naming)
    - [Submitting a pull request](#submitting-a-pull-request)
  - [Codebase structure](#codebase-structure)
  - [Developing LightAutoML](#developing-lightautoml)
    - [Installation](#installation)
      - [Optional Poetry workflow](#optional-poetry-workflow)
    - [Code conventions](#code-conventions)
    - [Tests and checks](#tests-and-checks)
      - [Automated code checking](#automated-code-checking)
      - [Testing machine-learning behavior](#testing-machine-learning-behavior)
  - [Preparing a release](#preparing-a-release)
    - [Required release benchmark](#required-release-benchmark)
    - [Publishing and verifying the release](#publishing-and-verifying-the-release)
  - [Documentation](#documentation)
    - [Building Documentation](#building-documentation)
    - [Writing Documentation](#writing-documentation)
    - [Adding new submodules](#adding-new-submodules)
    - [Adding Tutorials](#adding-tutorials)

## Contributing to LightAutoML

Thank you for your interest in contributing to LightAutoML! Before you begin writing code,
it is important that you share your intention to contribute with the developers team.

- First, please look for discussions on this topic in [issues](https://github.com/sb-ai-lab/LightAutoML/issues)
before implementing anything inside the project.
- Pick an issue and comment that you would like to work on it.
- If there is no discussion on this topic, create one.
  Please, include as much information as you can,
  any accompanying data (your tests, expected behavior, articles),
  and maybe your proposed solution.
- If you need more details, please ask we will provide them ASAP.

Follow the workflow below to prepare your branch and submit a pull request.

When adding functionality, please add examples that will fully explain it.
Examples can be added in several ways:

- [Inside the documentation](#writing-documentation)
- [Jupyter notebooks](#adding-tutorials)
- [Your own tests](#tests-and-checks)

### Fork and base branch

1. If you do not have write access to the repository, fork
   [LightAutoML](https://github.com/sb-ai-lab/LightAutoML) to your GitHub account.
   Clone your fork and add the original repository as `upstream`:

   ```bash
   git clone git@github.com:YOUR_GITHUB_USERNAME/LightAutoML.git
   cd LightAutoML
   git remote add upstream git@github.com:sb-ai-lab/LightAutoML.git
   ```

   Replace `YOUR_GITHUB_USERNAME` with your GitHub username. If you have write
   access, you may clone the original repository and use `origin` as upstream.

2. Use the base branch specified in the issue or agreed with the maintainers.
   If none is specified, ask which branch to use before starting work.
   Use the same branch as the target of your pull request.

3. Fetch the latest upstream changes and create a separate work branch using
   the naming rules below. For a fork, the commands are:

   ```bash
   git fetch upstream
   git switch -c WORK_BRANCH upstream/BASE_BRANCH
   ```

   Replace `WORK_BRANCH` with your new branch name and `BASE_BRANCH` with the
   agreed base branch. When working directly in the original repository,
   replace `upstream` with `origin` in these commands.

### Branch naming

Name work branches as:

```text
<type>/issue-<number>-<short-description>
```

Use lowercase kebab-case for `<short-description>`: keep it descriptive,
English, and brief. Choose one of these types:

- `feat` — new user-facing capability.
- `fix` — bug fix.
- `docs` — documentation-only change.
- `test` — test-only change.
- `refactor` — behavior-preserving restructuring.
- `chore` — tooling, CI, or maintenance work.
- `release` — version preparation or release maintenance.

Examples: `feat/issue-184-python-313-support`,
`fix/issue-271-handle-empty-target`, and
`docs/issue-305-installation-example`.

Do not use spaces, uppercase letters, opaque personal names, or generic names
such as `work` or `changes`. For a small change without an issue, omit the
issue segment (for example, `docs/update-contributing-guide`). Substantial
changes should be discussed in an issue before implementation, as described
above. Do not push branches named `experiment/*` to the main LightAutoML
repository.

### Submitting a pull request

1. Keep the change focused on the agreed issue. Add tests for changed behavior
   and documentation or examples for new functionality. Choose the checks in
   [Tests and checks](#tests-and-checks) that apply to your change and inspect
   `git diff` for unintended changes.
2. Commit your changes and push your work branch to your fork, or to the
   original repository if you have write access:

   ```bash
   git push -u origin WORK_BRANCH
   ```

3. Open a pull request against `sb-ai-lab/LightAutoML`. Select the agreed base
   branch as the target and your work branch as the source.
4. Start the description from the existing
   [pull request template](pull_request_template.md%E2%80%8E) and retain its headings:
   - `What was done` — describe the behavior changes and the validation commands
     and results. Explain any checks that were not run.
   - `Why / Goal` — explain the motivation and include the goal or milestone ID
     and title when applicable.
   - `Related issues` — reference the actual issue, for example `#123`, or use
     `N/A` for a small change without an issue.

   Remove placeholders and instructional comments before submitting.
5. Review the CI results and address failures and reviewer feedback. Push
   follow-up commits to the same work branch to update the pull request.

## Codebase structure

- [docs](../docs) - For documenting we use [Sphinx](https://www.sphinx-doc.org/).
  It provides easy to use auto-documenting via docstrings.
  - [Tutorial notebooks](../examples/tutorials) - Notebooks used in the documentation.
  - [Tutorial links](../docs/pages/tutorials) - Links that include the notebooks in Sphinx.

- [lightautoml](../lightautoml) - The code of LightAutoML library.
  - [addons](../lightautoml/addons) - Extensions of core functionality.
  - [automl](../lightautoml/automl) - The main module, which includes the AutoML class,
      blenders and ready-made presets.
  - [dataset](../lightautoml/dataset) - The internal interface for working with data.
  - [image](../lightautoml/image) - The internal interface for working with image data.
  - [ml_algo](../lightautoml/ml_algo) - Modules with machine learning algorithms
      and hyperparameters tuning tools.
  - [pipelines](../lightautoml/pipelines) - Pipelines for different tasks (feature processing & selection).
  - [reader](../lightautoml/reader) - Utils for training and analysing data.
  - [report](../lightautoml/report) - Report generators and templates.
  - [tasks](../lightautoml/tasks) - Define the task to solve its loss, metric.
  - [text](../lightautoml/text) - The internal interface for working with text data.
  - [transformers](../lightautoml/transformers) - Feature transformations.
  - [utils](../lightautoml/utils) - Common util tools (Timer, Profiler, Logging).
  - [validation](../lightautoml/validation) - Validation module.

## Developing LightAutoML

### Installation

Use Python 3.10 or later. If you have not cloned the project yet, follow
[Fork and base branch](#fork-and-base-branch). Run the commands below from the
`LightAutoML` repository root.

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, use these PowerShell commands instead:

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
```

Install pip updates and the tox version used by the test workflows:

```bash
python -m pip install --upgrade pip
python -m pip install "tox==3.28.0"
```

For documentation-only work or running checks through tox, this setup is
sufficient: tox installs its own dependencies. You can go directly to
[Tests and checks](#tests-and-checks) or
[Building Documentation](#building-documentation).

To develop library code or run examples in your active environment, also
install LightAutoML in editable mode:

```bash
python -m pip install -e .
```

Choose optional extras for the functionality you work on, for example:

```bash
python -m pip install -e ".[cv,nlp]"
```

Available extras are declared in [pyproject.toml](../pyproject.toml). Editable
installation makes source changes available in the environment without
reinstalling the library. Pip automatically installs the `poetry-core` build
backend in an isolated build environment; the Poetry CLI is not required.

#### Optional Poetry workflow

If you prefer Poetry, install it using
[the Poetry installation guide](https://python-poetry.org/docs/#installation)
and run `poetry install` from the repository root. Use `poetry run` to execute
commands in its environment. Install tox there as described above if needed.
The publishing workflow also uses Poetry.

### Code conventions

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) and the repository
  tooling configuration.
- Format Python with Black. The configured maximum line length is 120.
- Keep imports compatible with the configured isort profile: one import per
  line and two blank lines after the import block.
- Preserve public APIs unless the change explicitly requires a compatibility
  change. When changing an API, update its tests, documentation, and examples
  as applicable.
- Avoid broad refactors in a bug fix. Maintain existing naming, logging, type
  annotations, error handling, and data-interface conventions in the touched
  module.
- Do not add large datasets, model artifacts, generated documentation, or
  secrets to the repository.

### Tests and checks

Follow [Installation](#installation) to create a virtual environment and
install tox. Run the following commands from the repository root. Tox creates
isolated environments and installs each environment's dependencies from
[tox.ini](../tox.ini); installing LightAutoML in the outer environment is not
required to run these checks.

Choose checks according to the files and behavior you change:

| Change | Checks before a pull request |
| --- | --- |
| Markdown prose outside the Sphinx documentation | `lint` and `codespell`; no Python test matrix is needed. |
| Sphinx pages, docstrings, or tutorial notebooks | `lint`, `codespell`, and `docs`; run relevant tests too if code behavior changes. |
| Library code or tests | `lint`, `codespell`, focused tests, then the full test suite for an available supported Python version; add `docs` for public API or documentation changes. |
| Dependencies, packaging, hooks, or CI configuration | `lint`, `codespell`, and the affected tox environments; check the supported Python versions affected by the change. |

Add or update tests for every behavior change. Start with the narrowest
relevant check, then broaden it as indicated above:

```bash
# One test module or directory; choose an installed supported Python environment
tox -e py312 -- -x tests/unit/test_utils

# All tests for that Python version
tox -e py312

# Formatting, linting, and repository hooks
tox -e lint

# Spelling
tox -e codespell

# Documentation
tox -e docs

# Optional complete local matrix: all configured Python versions and checks
tox --parallel 6
```

Replace `py312` with an available environment declared in `tox.ini`: `py310`,
`py311`, `py312`, `py313`, or `py314`. Each test environment requires its
matching Python interpreter. The optional complete local matrix requires all
of these interpreters; running tox does not install them. A documentation-only
change does not require this matrix. Do not claim that an unavailable Python
version was tested.

The test environments install the `cv`, `nlp`, and `afg` extras and run pytest
with warnings treated as errors (apart from the explicit exceptions in
`tox.ini`). Do not suppress a new warning merely to make a test pass; fix its
cause or explain why it is unavoidable. If a full test suite cannot be run,
state exactly what was run and why the remaining checks were skipped.

#### Automated code checking

We use [pre-commit](https://pre-commit.com/) to automate code quality checks.
The `lint` environment installs a Git commit hook and runs the checks on all
tracked files. The hook set includes Black, Flake8, trailing-whitespace and
EOF checks, debug-statement checks, YAML validation, and rstcheck. Spelling
is checked separately by the `codespell` environment.

Once installed, the commit hook runs on `git commit`. Hooks can modify files.
Always inspect `git status` and `git diff` afterward and keep only changes
relevant to the task. If a hook fixes files, review and
stage those changes, then retry the commit. If a check fails without fixing
the issue, read its message, fix the cause, and retry. Skipping the commit hook
with `git commit --no-verify` does not bypass the CI checks.

#### Testing machine-learning behavior

- Prefer small deterministic synthetic fixtures and set `random_state` or all
  applicable random seeds.
- Avoid external network access, dataset or model downloads, GPU requirements,
  and unnecessarily expensive training in unit tests.
- Preserve index alignment, feature names, shapes, dtypes, sparse/dense
  behavior, and train/validation separation where those are part of the API.
- Compare floating-point results with suitable tolerances instead of exact
  equality.

## Preparing a release

Release preparation is a maintainer responsibility. Agree on the release
branch, version, and scope, then prepare a focused release pull request using
the [branch naming rules](#branch-naming).

Before tagging the release:

1. Update `tool.poetry.version` in [pyproject.toml](../pyproject.toml).
   `lightautoml.__version__` is read from installed package metadata and does
   not need a separate edit. Check the supported Python range, dependencies,
   and extras.
2. Prepare release notes covering features, fixes, dependency changes,
   compatibility changes, and any migration steps.
3. Obtain passing results for the supported Python test matrix, `lint`,
   `codespell`, and `docs` as described in [Tests and checks](#tests-and-checks).
   Unlike an ordinary documentation change, a release needs the full matrix;
   it may be run in CI instead of on one local machine.
4. Build the wheel and source distribution. Check their version, metadata,
   and contents, and verify installation and a small training/prediction
   example in clean environments for both distributions.
5. Complete the [required release benchmark](#required-release-benchmark)
   against the previous stable release and attach the comparison to the
   release pull request. Unit tests alone do not establish that model quality
   has been preserved.

Record the exact candidate commit tested and benchmarked. Changes affecting
code, dependencies, configuration, or packaging require a new candidate and
the affected checks and benchmarks to be repeated before publication.

### Required release benchmark

Compare the exact release candidate with the previous stable release on the
agreed benchmark suite under comparable conditions. Review quality per
dataset and training/inference time, using acceptance criteria agreed before
the runs. Require complete results and resolve confirmed regressions before
publication.

Follow the team's
[release benchmark instructions in experimenttracking](https://tech-point.ailabtools.org/ai-lab-pmo/mltools/automl/experimenttracking/-/blob/master/docs/release-benchmark.md)
for setup, execution, and comparison. Attach a concise benchmark conclusion
to the release pull request and link to the detailed report in the appropriate
workspace. Keep infrastructure details and internal experiment data there.

### Publishing and verifying the release

After the release pull request is approved and merged, verify that the
release commit matches the accepted checks and benchmark evidence. Agree on
the tag spelling and ensure its version matches `pyproject.toml`; historical
tags use both `v0.x.y` and `v.0.x.y`.

Pushing a tag matching `v*.*.*` triggers
[publish_pypi.yml](workflows/publish_pypi.yml), which publishes using
`LAMA_PYPI_TOKEN`. The publishing job does not wait for tests or benchmark
acceptance; maintainers must complete the checks above before pushing the tag.

After publication, check the workflow result, install the published version
from PyPI in a clean environment, and verify its version and a small
training/prediction example. Publish a GitHub Release with the release notes
and a summary of the accepted benchmark comparison.

## Documentation

Build the documentation locally after changing Sphinx pages, docstrings,
public APIs, or tutorials. The documentation uses Sphinx and is published on
Read the Docs. Its configuration in [docs/conf.py](../docs/conf.py) mocks heavy
library dependencies so the documentation can be built without installing
the full machine-learning environment.

### Building Documentation

Follow [Fork and base branch](#fork-and-base-branch) if you have not cloned the
repository yet. Install tox as described in [Tests and checks](#tests-and-checks)
and ensure that `make` is available on your `PATH` (GNU Make on Windows).
From the repository root, run:

```bash
tox -e docs
```

The `docs` environment installs Sphinx, its theme and extensions, and
`pypandoc-binary`, which supplies Pandoc for notebook conversion. Installing
the `cv` and `nlp` extras alone does not install these documentation tools.
The dependency list in [docs/requirements.txt](../docs/requirements.txt) is
shared by tox and Read the Docs. The GitHub Actions documentation workflow
runs the same tox environment; Read the Docs installs these requirements with
pip and uses its standard Sphinx build without installing the ML dependencies.

Tox runs these steps in order:

1. `make clean html` in `docs/` removes previous build output and generates
   the HTML documentation.
2. `python ../check_docs.py` checks for the HTML files expected by the
   repository's post-build validation script. It does not generate HTML.

Open `docs/_build/html/index.html` to preview the result and inspect the build
warnings for pages affected by your change. Do not commit generated HTML.

### Writing Documentation

There are some rules, that docstrings should fit.

1. LightAutoML uses [Google-style docstring formatting](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
   The length of the line inside docstring should be limited
   to 80 characters to fit into Jupyter documentation popups.

2. Every non-one-line docstring should have a paragraph at its end, regardless of where it will be used:
   in the documentation for a class, module, function, class
   method, etc. One-liners or descriptions,
   that have no special directives (Args, Warning, Note, etc.) may have no paragraph at its end.

3. Once you added some module to LightAutoML,
   you should add some info about it at the beginning of the module.
   An example is available in [docs/mock_docs.py](../docs/mock_docs.py).
   Also, if you use submodules, please add description to `__init__.py`
   (it is usefull for Sphinx's autosummary).

4. Please use references to other submodules. You can do it by Sphinx directives.
   For more information: <https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html>
5. There is an example for documenting standalone functions.

```python3
from typing import List, Union

import numpy as np
import torch

def typical_function(a: int, b: Union['np.ndarray', None] = None) -> List[int]:
    """Short function description, terminated by dot.

    Some details. The parameter after arrow is return value type.

    Use 2 newlines to make a new paragraph,
    like in `LaTeX by Knuth <https://en.wikipedia.org/wiki/LaTeX>`_.

    Args:
        a: Parameter description, starting with a capital
          latter and terminated by a period.
        b: Textual parameter description.

    .. note::
        Some additional notes, with special block.

        If you want to itemize something (it is inside note):

            - First option.
            - Second option.
              Just link to function :func:`torch.cuda.current_device`.
            - Third option.
              Also third option.
            - It will be good if you don't use it in args.

    Warning:
        Some warning. Every block should be separated
        with other block with paragraph.

    Warning:
        One more warning. Also notes and warnings
        can be upper in the long description of function.

    Example:

        >>> print('MEME'.lower())
        meme
        >>> b = typical_function(1, np.ndarray([1, 2, 3]))

    Returns:
        Info about return value.

    Raises:
        Exception: Exception description.

    """

    return [a, 2, 3]
```

6. Docstring for generator function.

```python3
def generator_func(n: int):
    """Generator have a ``Yields`` section instead of ``Returns``.

    Args:
        n: Number of interations.

    Yields:
        The next number in the range of ``0`` to ``n-1``.

    Example:
        Example description.

        >>> print([i for i in generator_func(4)])
        [0, 1, 2, 3]

    """
    x = 0
    while x < n:
        yield x
        x += 1
```

7. Documenting classes.

```python3
from typing import List, Union
import numpy as np
import torch


class ExampleClass:
    """The summary line for a class that fits only one line.

    Long description.

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section, like in ``Args`` section of function.

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method. Use arrow to set the return type.

    On the stage before __init__ we don't know anything about `Attributes`,
    so please, add description about it's types.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """

    def __init__(self, param1: int, param2: 'np.ndarray', *args, **kwargs):
        """Example of docstring of the __init__ method.

        Note:
            You can also add notes as ``Note`` section.
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: Description of `param1`.
            param2: Description of `param2`.
            *args: Description of positional arguments.
            **kwargs: Description of key-word arguments.

        """
        self.attr1 = param1
        self.attr2 = param2
        if len(args) > 0:
            self.attr2 = args[0]
        self.attr3 = kwargs # will not be documented.
        self.figure = 4 * self.attr1

    @property
    def readonly_property(self) -> str:
        """Properties should be documented in
        their getter method.

        """
        return 'lol'

    @property
    def readwrite_property(self) -> List[str]:
        """Properties with both a getter and setter
        should only be documented in their getter method.

        If the setter method contains notable behavior, it should be
        mentioned here.
        """
        return [str(self.figure)]

    @readwrite_property.setter
    def readwrite_property(self, value: int):
        self.figure = value

    def some_method(self, param1: int, param2: float = np.pi) -> List[int]:
        """Just like a functions.

        Long description.

        Warning:
            This method do something. May be undefined-behaviour.

        Args:
            param1: Some description of param1.
            param2: Some description of param2. Default value
               will be contained in signature of function.

        Returns:
            Array with `1`, `2`, `3`.

        """
        self.attr1 = param1
        self.attr2 += param2

        return [1, 2, 3]


    def __special__(self):
        """By default we aren`t include dundered members.

        Also there may be no docstring.
        """
        pass

    def _private(self):
        """By default we aren't include private members.

        Also there may be no docstring.
        """
        pass

    @staticmethod
    def static_method(param1: int):
        """Description of static method.

        Note:
            As like common method of class don`t use `self`.

        Args:
            param1: Description of `param1`.

        """
        print(param1)
```

8. If you have a parameter that can take a finite number of values,
   if possible, describe each of them in the Note section.

```python3
import random


class A:
    """
    Some description.

    Some long description.

    Attributes:
        attr1 (:obj:`int`): Description of `attr1`.
        attr2 (:obj:`int`): Description of `attr2`.

    """
    def __init__(self, weight_initialization: str = 'none'):
        """

        Args:
            weight_initialization: Initialization type.

        Note:
            There are several initialization types:

                - '`zeros`': fill ``attr1``
                  and ``attr2`` with zeros.
                - '`ones`': fill ``attr1``
                  and ``attr2`` with ones.
                - '`none`': fill ``attr1``
                  and ``attr2`` with random int in `\[0, 100\]`.

        Raises:
            ValueError: If the entered initialization type is not supported.

        """
        if weight_initialization not in ['zeros', 'ones', 'none']:
            raise ValueError(
                f'{weight_initialization} - Unsupported weight initialization.')

        if weight_initialization == 'zeros':
            attr1 = 0
            attr2 = 0
        elif weight_initialization == 'ones':
            attr1 = 1
            attr2 = 1
        else:
            attr1 = random.randint(0, 100)
            attr2 = random.randint(0, 100)
```

### Adding new submodules

When adding a library submodule, create or update its `.rst` page in
[docs/pages/modules](../docs/pages/modules). Register a new module page in the
toctree in [docs/pages/Python-API.rst](../docs/pages/Python-API.rst).

Add new classes and functions to the corresponding module page. For a new
top-level documentation page, update the toctree in
[docs/index.rst](../docs/index.rst). Run the
[documentation build](#building-documentation) to check the result.

### Adding Tutorials

We use [nbsphinx](https://nbsphinx.readthedocs.io/) and nbsphinx-link to include
tutorial notebooks:

1. Put the notebook in [examples/tutorials](../examples/tutorials). New
   notebooks match the repository's `*.ipynb` ignore rule; explicitly stage
   the intended notebook with `git add -f examples/tutorials/YOUR_NOTEBOOK.ipynb`.
2. Add a `.nblink` file in [docs/pages/tutorials](../docs/pages/tutorials),
   pointing to the notebook relative to that directory. Follow the existing
   [Tutorial_1_basics.nblink](../docs/pages/tutorials/Tutorial_1_basics.nblink)
   example, including any required media paths.
3. Add the `.nblink` file to the appropriate toctree in
   [docs/pages/Tutorials.rst](../docs/pages/Tutorials.rst).
4. Run the notebook to check its examples and save its intended output, then
   run the [documentation build](#building-documentation). The Sphinx
   configuration uses `nbsphinx_execute = "never"`, so building the
   documentation does not execute notebook cells.
