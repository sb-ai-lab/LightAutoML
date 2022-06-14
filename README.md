# LightAutoML - automatic model creation framework

[![Slack](https://lightautoml-slack.herokuapp.com/badge.svg)](https://lightautoml-slack.herokuapp.com)
[![Telegram](https://img.shields.io/badge/chat-on%20Telegram-2ba2d9.svg)](https://t.me/lightautoml)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lightautoml?color=green&label=PyPI%20downloads&logo=pypi&logoColor=orange&style=plastic)
![Read the Docs](https://img.shields.io/readthedocs/lightautoml?style=plastic)

LightAutoML project from Sberbank AI Lab AutoML group is the framework for automatic classification and regression model creation.

Current available tasks to solve:
- binary classification
- multiclass classification
- regression

Currently we work with datasets, where **each row is an object with its specific features and target**. Multitable datasets and sequences are now under contruction :)

**Note**: for automatic creation of interpretable models we use [`AutoWoE`](https://github.com/sberbank-ai-lab/AutoMLWhitebox) library made by our group as well.

**Authors**: [Alexander Ryzhkov](https://kaggle.com/alexryzhkov), [Anton Vakhrushev](https://kaggle.com/btbpanda), [Dmitry Simakov](https://kaggle.com/simakov), Vasilii Bunakov, Rinchin Damdinov, Pavel Shvets, Alexander Kirilin

**LightAutoML video guides**:
- (Russian) [LightAutoML webinar for Sberloga community](https://www.youtube.com/watch?v=ci8uqgWFJGg) ([Alexander Ryzhkov](https://kaggle.com/alexryzhkov), [Dmitry Simakov](https://kaggle.com/simakov))
- (Russian) [LightAutoML hands-on tutorial in Kaggle Kernels](https://www.youtube.com/watch?v=TYu1UG-E9e8) ([Alexander Ryzhkov](https://kaggle.com/alexryzhkov))
- (English) [Automated Machine Learning with LightAutoML: theory and practice](https://www.youtube.com/watch?v=4pbO673B9Oo) ([Alexander Ryzhkov](https://kaggle.com/alexryzhkov))
- (English) [LightAutoML framework general overview, benchmarks and advantages for business](https://vimeo.com/485383651) ([Alexander Ryzhkov](https://kaggle.com/alexryzhkov))
- (English) [LightAutoML practical guide - ML pipeline presets overview](https://vimeo.com/487166940) ([Dmitry Simakov](https://kaggle.com/simakov))

**Articles about LightAutoML:**
- (English) [LightAutoML vs Titanic: 80% accuracy in several lines of code (Medium)](https://alexmryzhkov.medium.com/lightautoml-preset-usage-tutorial-2cce7da6f936)
- (English) [Hands-On Python Guide to LightAutoML – An Automatic ML Model Creation Framework (Analytic Indian Mag)](https://analyticsindiamag.com/hands-on-python-guide-to-lama-an-automatic-ml-model-creation-framework/?fbclid=IwAR0f0cVgQWaLI60m1IHMD6VZfmKce0ZXxw-O8VRTdRALsKtty8a-ouJex7g)

See the [Documentation of LightAutoML](https://lightautoml.readthedocs.io/).

*******
# Installation
### Installation via pip from PyPI
To install LAMA framework on your machine:
```bash
pip install -U lightautoml
```
### Installation from sources with virtual environment creation
If you want to create a specific virtual environment for LAMA, you need to install  `python3-venv` system package and run the following command, which creates `lama_venv` virtual env with LAMA inside:
```bash
bash build_package.sh
```
To check this variant of installation and run all the demo scripts, use the command below:
```bash
bash test_package.sh
```
To install optional support for generating reports in pdf format run following commands:
```bash
# MacOS
brew install cairo pango gdk-pixbuf libffi

# Debian / Ubuntu
sudo apt-get install build-essential libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info

# Fedora
sudo yum install redhat-rpm-config libffi-devel cairo pango gdk-pixbuf2

# Windows
# follow this tutorial https://weasyprint.readthedocs.io/en/stable/install.html#windows

poetry install -E pdf
```
*******
# Docs generation
```bash
bash build_docs.sh
```

Builded official documentation for LightAutoML is available [`here`](https://lightautoml.readthedocs.io/en/latest/).
*******
# Usage examples

To find out how to work with LightAutoML, we have several [`tutorials`](examples/). You can run them in Google Colab:
1. `Tutorial_1. Create your own pipeline.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sberbank-ai-lab/LightAutoML/blob/master/Tutorial_1.%20Create%20your%20own%20pipeline.ipynb) - shows how to create your own pipeline from specified blocks: pipelines for feature generation and feature selection, ML algorithms, hyperparameter optimization etc.
2. `Tutorial_2. AutoML pipeline preset.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sberbank-ai-lab/LightAutoML/blob/master/Tutorial_2.%20AutoML%20pipeline%20preset.ipynb) - shows how to use LightAutoML presets (both standalone and time utilized variants) for solving ML tasks on tabular data. Using presets you can solve binary classification, multiclass classification and regression tasks, changing the first argument in Task.
3. `Tutorial_3. Multiclass task.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sberbank-ai-lab/LightAutoML/blob/master/Tutorial_3.%20Multiclass%20task.ipynb) - shows how to build ML pipeline for multiclass ML task by hand
4. `Tutorial_4. SQL data source for pipeline preset.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sberbank-ai-lab/LightAutoML/blob/master/Tutorial_4.%20SQL%20data%20source%20for%20pipeline%20preset.ipynb) - shows how to use LightAutoML presets (both standalone and time utilized variants) for solving ML tasks on tabular data from SQL data base instead of CSV.

Each tutorial has the step to enable Profiler and completes with Profiler run, which generates distribution for each function call time and shows it in interactive HTML report: the report show full time of run on its top and interactive tree of calls with percent of total time spent by the specific subtree.

**Important 1**: for production you have no need to use profiler (which increase work time and memory consomption), so please do not turn it on - it is in off state by default

**Important 2**: to take a look at this report after the run, please comment last line of demo with report deletion command.

Kaggle kernel examples of LightAutoML usage:
- [Tabular Playground Series April 2021 competition solution](https://www.kaggle.com/alexryzhkov/n3-tps-april-21-lightautoml-starter)
- [Titanic competition solution (80% accuracy)](https://www.kaggle.com/alexryzhkov/lightautoml-titanic-love)
- [Titanic **12-code-lines** competition solution (78% accuracy)](https://www.kaggle.com/alexryzhkov/lightautoml-extreme-short-titanic-solution)
- [House prices competition solution](https://www.kaggle.com/alexryzhkov/lightautoml-houseprices-love)
- [Natural Language Processing with Disaster Tweets solution](https://www.kaggle.com/alexryzhkov/lightautoml-starter-nlp)
- [Tabular Playground Series March 2021 competition solution](https://www.kaggle.com/alexryzhkov/lightautoml-starter-for-tabulardatamarch)
- [Tabular Playground Series February 2021 competition solution](https://www.kaggle.com/alexryzhkov/lightautoml-tabulardata-love)
- [Interpretable WhiteBox solution](https://www.kaggle.com/simakov/lama-whitebox-preset-example)
- [Custom ML pipeline elements inside existing ones](https://www.kaggle.com/simakov/lama-custom-automl-pipeline-example)

For more examples, in `tests` folder you can find different scenarios of LightAutoML usage:
1. `demo0.py` - building ML pipeline from blocks and fit + predict the pipeline itself.
2. `demo1.py` - several ML pipelines creation (using importances based cutoff feature selector) to build 2 level stacking using AutoML class
3. `demo2.py` - several ML pipelines creation (using iteartive feature selection algorithm) to build 2 level stacking using AutoML class
4. `demo3.py` - several ML pipelines creation (using combination of cutoff and iterative FS algos) to build 2 level stacking using AutoML class
5. `demo4.py` - creation of classification and regression tasks for AutoML with loss and evaluation metric setup
6. `demo5.py` - 2 level stacking using AutoML class with different algos on first level including LGBM, Linear and LinearL1
7. `demo6.py` - AutoML with nested CV usage
8. `demo7.py` - AutoML preset usage for tabular datasets (predefined structure of AutoML pipeline and simple interface for users without building from blocks)
9. `demo8.py` - creation pipelines from blocks to build AutoML, solving multiclass classification task
10. `demo9.py` - AutoML time utilization preset usage for tabular datasets (predefined structure of AutoML pipeline and simple interface for users without building from blocks)
11. `demo10.py` - creation pipelines from blocks (including CatBoost) to build AutoML, solving multiclass classification task
12. `demo11.py` - AutoML NLP preset usage for tabular datasets with text columns
13. `demo12.py` - AutoML tabular preset usage with custom validation scheme and multiprocessed inference


******
# Contributing to LightAutoML

If you are interested in contributing to LightAutoML, please read the [Contributing Guide](.github/CONTRIBUTING.md) to get started.


*******
# Questions / Issues / Suggestions

Write a message to us:
- [Alexander Ryzhkov](https://kaggle.com/alexryzhkov) (_email_: AMRyzhkov@sberbank.ru, _telegram_: @RyzhkovAlex)
- [Anton Vakhrushev](https://kaggle.com/btbpanda) (_email_: AGVakhrushev@sberbank.ru)
- [Dmitry Simakov](https://kaggle.com/simakov) (_email_: Simakov.D.E@sberbank.ru)
