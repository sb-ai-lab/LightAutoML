# Table of contents

- [Contributing to LightAutoML](#contriburing-to-lightautoml)
- [Codebase Structure](#codebase-structure)
- [Developing LightAutoML](#developing-lightautoml)
- [Writing Documentation](#writing-documentation)
- [Style Guide](#style-guide)

## Contributing to LightAutoML

Thank you for your interest in contributing to LightAutoML! Before you begin writing code,
it is important that you share your intention to contribute with the developers team.

- First, please look for discussions on this topic in [issues](https://github.com/sberbank-ai-lab/LightAutoML/issues)
before implementing anything inside the project.
- Pick an issue and comment that you would like to work on it.
- If there is no discussion on this topic, create one.
  Please, include as much information as you can,
  any accompanying data (your tests, expected behavior, articles),
  and maybe your proposed solution.
- If you need more details, please ask we will provide them ASAP.

Once you implement and test your feature or bug-fix, please submit
a Pull Request to https://github.com/sberbank-ai-lab/LightAutoML.

When adding functionality, please add examples that will fully explain it.
Examples can be added in several ways:
- [Inside the documentation](#writing-documentation)
- [Jupyter notebooks](#adding-tutorials)
- [Your own tests](#testing)

## Codebase structure

- [docs](docs) - For documenting we use [Sphinx](https://www.sphinx-doc.org/).
  It provides easy to use auto-documenting via docstrings.
  - [Tutorials](docs/tutorials) - Notebooks with tutorials.

- [lightautoml](lightautoml) - The code of LightAutoML library.
    - [addons](lightautoml/addons) - Extensions of core functionality.
    - [automl](lightautoml/automl) - The main module, which includes the AutoML class,
      blenders and ready-made presets.
    - [dataset](lightautoml/dataset) - The internal interface for working with data.
    - [image](lightautoml/image) - The internal interface for working with image data.
    - [ml_algo](lightautoml/ml_algo) - Modules with machine learning algorithms
      and hyperparameters tuning tools.
    - [pipelines](lightautoml/pipelines) - Pipelines for different tasks (feature processing & selection).
    - [reader](lightautoml/reader) - Utils for training and analysing data.
    - [report](lightautoml/report) - Report generators and templates.
    - [tasks](lightautoml/tasks) - Define the task to solve its loss, metric.
    - [text](lightautoml/text) - The internal interface for working with text data.
    - [transformers](lightautoml/transformers) - Feature transformations.
    - [utils](lightautoml/utils) - Common util tools (Timer, Profiler, Logging).
    - [validation](lightautoml/validation) - Validation module.


## Developing LightAutoML

### Installation

If you are installing from the source, you will need Python 3.6.12 or later.
We recommend you install an [Anaconda](https://www.anaconda.com/products/individual#download-section)
to work with environments.


1. Install poetry using [the poetry installation guide](https://python-poetry.org/docs/#installation).

2. Clone the project to your own local machine:
```bash
git clone git@github.com:sberbank-ai-lab/LightAutoML.git
cd LightAutoML
```

3. Install LightAutoML:
```bash
poetry install
```

After that, there is virtual environment, where you can test and implement your own code.
So, you don't need to rebuild the full project every time.
Each change in the code will be reflected in the library inside the environment.


### Style Guide

We follow [the standard python PEP8](https://www.python.org/dev/peps/pep-0008/) conventions for style.

#### Automated code checking

In order to automate checking of the code quality, we use
[pre-commit](https://pre-commit.com/). For more details, see the documentation,
here we will give a quick-start guide:
1. Install and configure:
```console
poetry run pre-commit install
```
2. Now, when you run `$ git commit`, there will be a pre-commit check.
   This is going to search for issues in your code: spelling, formatting, etc.
   In some cases, it will automatically fix the code, in other cases, it will
   print a warning. If it automatically fixed the code, you'll need to add the
   changes to the index (`$ git add FILE.py`) and run `$ git commit` again. If
   it didn't automatically fix the code, but still failed, it will have printed
   a message as to why the commit failed. Read the message, fix the issues,
   then recommit.
3. The pre-commit checks are done to avoid pushing and then failing. But, you
   can skip them by running `$ git commit --no-verify`, but note that the C.I.
   still does the check so you won't be able to merge until the issues are
   resolved.
If you experience any issues with pre-commit, please ask for support on the
usual help channels.

### Testing

Before making a pull request (despite changing only the documentation or writing new code), please check your code on tests:
```bash
poetry run pytest tests
```

To run tests with different Python versions, run tox
```bash
poetry run tox
```
Also if you develop new functionality, please add your own tests.


## Documentation


Before writing the documentation, you should collect it to make sure that the code
you wrote doesn't break the rest of the documentation. The library might work,
but the documentation might not be. It is built on the Read the Docs service,
which uses its own virtual environment, which contains only part
of the LightAutoML library dependencies. This is done to make
the documentation more lightweight.

By default, functions, that have no description will be mock from overall documentation.

### Building Documentation:

To build the documentation:

1. Clone repository to your device.
```
git clone https://github.com/sberbank-ai-lab/LightAutoML
cd LightAutoML
```


2. Make environment and install requirements.
```bash
poetry install -E cv -E nlp
```

3. Remove existing html files:
```bash
cd docs
poetry run make clean html
cd ..
```

4. Generate HTML documentation files. The generated files will be in `docs/_build/html`.
```bash
poetry run python check_docs.py
```


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
   Example of this you can find in `docs/mock_docs.py`.
   Also, if you use submodules, please add description to `__init__.py`
   (it is usefull for Sphinx's autosummary).

4. Please use references to other submodules. You can do it by Sphinx directives.
   For more information: https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html
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

If you add your own directory to LightAutoML, you should add a corresponding module as new `.rst`
file to the `docs/`.  And also mention it in `docs/index.rst`.

If you add your own module, class or function, then you will need
to add it description to the corresponding `.rst` in `docs`.


### Adding Tutorials

We use [nbsphinx](https://nbsphinx.readthedocs.io/) extension for tutorials.
Examples, you can find in `docs/notebooks`.
Please, put your tutorial in this folder
and after add it in `docs/Tutorials.rst`.
