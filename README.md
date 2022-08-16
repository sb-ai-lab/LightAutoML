<img src=https://github.com/AILab-MLTools/LightAutoML/raw/master/imgs/LightAutoML_logo_big.png />

# LightAutoML - automatic model creation framework

[![Telegram](https://img.shields.io/badge/chat-on%20Telegram-2ba2d9.svg)](https://t.me/lightautoml)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lightautoml?color=green&label=PyPI%20downloads&logo=pypi&logoColor=orange&style=plastic)
![Read the Docs](https://img.shields.io/readthedocs/lightautoml?style=plastic)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

LightAutoML (LAMA) is an AutoML framework which provides automatic model creation for the following tasks:
- binary classification
- multiclass  classification
- regression

Current version of the package handles datasets that have independent samples in each row. I.e. **each row is an object with its specific features and target**.
Multitable datasets and sequences are a work in progress :)

**Note**: we use [`AutoWoE`](https://pypi.org/project/autowoe) library to automatically create interpretable models.

**Authors**: [Alexander Ryzhkov](https://kaggle.com/alexryzhkov), [Anton Vakhrushev](https://kaggle.com/btbpanda), [Dmitry Simakov](https://kaggle.com/simakov), Vasilii Bunakov, Rinchin Damdinov, Alexander Kirilin, Pavel Shvets.

**Documentation** of LightAutoML is available [here](https://lightautoml.readthedocs.io/), you can also [generate](https://github.com/AILab-MLTools/LightAutoML/blob/master/.github/CONTRIBUTING.md#building-documentation) it.

# (New features) GPU and Spark pipelines
Full GPU and Spark pipelines for LightAutoML currently available for developers testing (still in progress). The code and tutorials for:
- GPU pipeline is [available here](https://github.com/Rishat-skoltech/LightAutoML_GPU)
- Spark pipeline is [available here](https://github.com/sb-ai-lab/SLAMA)

<a name="toc"></a>
# Table of Contents

* [Installation LightAutoML from PyPI](#installation)
* [Quick tour](#quicktour)
* [Resources](#examples)
* [Contributing to LightAutoML](#contributing)
* [License](#apache)
* [For developers](#developers)
* [Support and feature requests](#support)

<a name="installation"></a>
# Installation
To install LAMA framework on your machine from PyPI, execute following commands:
```bash

# Install base functionality:

pip install -U lightautoml

# For partial installation use corresponding option.
# Extra dependecies: [nlp, cv, report]
# Or you can use 'all' to install everything

pip install -U lightautoml[nlp]

```

Additionaly, run following commands to enable pdf report generation:

```bash
# MacOS
brew install cairo pango gdk-pixbuf libffi

# Debian / Ubuntu
sudo apt-get install build-essential libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info

# Fedora
sudo yum install redhat-rpm-config libffi-devel cairo pango gdk-pixbuf2

# Windows
# follow this tutorial https://weasyprint.readthedocs.io/en/stable/install.html#windows
```
[Back to top](#toc)

<a name="quicktour"></a>
# Quick tour

Let's solve the popular Kaggle Titanic competition below. There are two main ways to solve machine learning problems using LightAutoML:
* Use ready preset for tabular data:
```python
import pandas as pd
from sklearn.metrics import f1_score

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

df_train = pd.read_csv('../input/titanic/train.csv')
df_test = pd.read_csv('../input/titanic/test.csv')

automl = TabularAutoML(
    task = Task(
        name = 'binary',
        metric = lambda y_true, y_pred: f1_score(y_true, (y_pred > 0.5)*1))
)
oof_pred = automl.fit_predict(
    df_train,
    roles = {'target': 'Survived', 'drop': ['PassengerId']}
)
test_pred = automl.predict(df_test)

pd.DataFrame({
    'PassengerId':df_test.PassengerId,
    'Survived': (test_pred.data[:, 0] > 0.5)*1
}).to_csv('submit.csv', index = False)
```

LighAutoML framework has a lot of ready-to-use parts and extensive customization options, to learn more check out the [resources](#Resources) section.

[Back to top](#toc)

<a name="examples"></a>
# Resources

### Kaggle kernel examples of LightAutoML usage:

- [Tabular Playground Series April 2021 competition solution](https://www.kaggle.com/alexryzhkov/n3-tps-april-21-lightautoml-starter)
- [Titanic competition solution (80% accuracy)](https://www.kaggle.com/alexryzhkov/lightautoml-titanic-love)
- [Titanic **12-code-lines** competition solution (78% accuracy)](https://www.kaggle.com/alexryzhkov/lightautoml-extreme-short-titanic-solution)
- [House prices competition solution](https://www.kaggle.com/alexryzhkov/lightautoml-houseprices-love)
- [Natural Language Processing with Disaster Tweets solution](https://www.kaggle.com/alexryzhkov/lightautoml-starter-nlp)
- [Tabular Playground Series March 2021 competition solution](https://www.kaggle.com/alexryzhkov/lightautoml-starter-for-tabulardatamarch)
- [Tabular Playground Series February 2021 competition solution](https://www.kaggle.com/alexryzhkov/lightautoml-tabulardata-love)
- [Interpretable WhiteBox solution](https://www.kaggle.com/simakov/lama-whitebox-preset-example)
- [Custom ML pipeline elements inside existing ones](https://www.kaggle.com/simakov/lama-custom-automl-pipeline-example)

### Google Colab tutorials and [other examples](examples/):

- [`Tutorial_1_basics.ipynb`](https://colab.research.google.com/github/AILab-MLTools/LightAutoML/blob/master/examples/tutorials/Tutorial_1_basics.ipynb) - get started with LightAutoML on tabular data.
- [`Tutorial_2_WhiteBox_AutoWoE.ipynb`](https://colab.research.google.com/github/AILab-MLTools/LightAutoML/blob/master/examples/tutorials/Tutorial_2_WhiteBox_AutoWoE.ipynb) - creating interpretable models.
- [`Tutorial_3_sql_data_source.ipynb`](https://colab.research.google.com/github/AILab-MLTools/LightAutoML/blob/master/examples/tutorials/Tutorial_3_sql_data_source.ipynb) - shows how to use LightAutoML presets (both standalone and time utilized variants) for solving ML tasks on tabular data from SQL data base instead of CSV.
- [`Tutorial_4_NLP_Interpretation.ipynb`](https://colab.research.google.com/github/AILab-MLTools/LightAutoML/blob/master/examples/tutorials/Tutorial_4_NLP_Interpretation.ipynb) - example of using TabularNLPAutoML preset, LimeTextExplainer.
- [`Tutorial_5_uplift.ipynb`](https://colab.research.google.com/github/AILab-MLTools/LightAutoML/blob/master/examples/tutorials/Tutorial_5_uplift.ipynb) - shows how to use LightAutoML for a uplift-modeling task.
- [`Tutorial_6_custom_pipeline.ipynb`](https://colab.research.google.com/github/AILab-MLTools/LightAutoML/blob/master/examples/tutorials/Tutorial_6_custom_pipeline.ipynb) - shows how to create your own pipeline from specified blocks: pipelines for feature generation and feature selection, ML algorithms, hyperparameter optimization etc.
- [`Tutorial_7_ICE_and_PDP_interpretation.ipynb`](https://colab.research.google.com/github/AILab-MLTools/LightAutoML/blob/master/examples/tutorials/Tutorial_7_ICE_and_PDP_interpretation.ipynb) - shows how to obtain local and global interpretation of model results using ICE and PDP approaches.

**Note 1**: for production you have no need to use profiler (which increase work time and memory consomption), so please do not turn it on - it is in off state by default

**Note 2**: to take a look at this report after the run, please comment last line of demo with report deletion command.

### Courses, videos and papers

* **LightAutoML crash courses**:
    - (Russian) [AutoML course for OpenDataScience community](https://ods.ai/tracks/automl-course-part1)

* **Video guides**:
    - (Russian) [LightAutoML webinar for Sberloga community](https://www.youtube.com/watch?v=ci8uqgWFJGg) ([Alexander Ryzhkov](https://kaggle.com/alexryzhkov), [Dmitry Simakov](https://kaggle.com/simakov))
    - (Russian) [LightAutoML hands-on tutorial in Kaggle Kernels](https://www.youtube.com/watch?v=TYu1UG-E9e8) ([Alexander Ryzhkov](https://kaggle.com/alexryzhkov))
    - (English) [Automated Machine Learning with LightAutoML: theory and practice](https://www.youtube.com/watch?v=4pbO673B9Oo) ([Alexander Ryzhkov](https://kaggle.com/alexryzhkov))
    - (English) [LightAutoML framework general overview, benchmarks and advantages for business](https://vimeo.com/485383651) ([Alexander Ryzhkov](https://kaggle.com/alexryzhkov))
    - (English) [LightAutoML practical guide - ML pipeline presets overview](https://vimeo.com/487166940) ([Dmitry Simakov](https://kaggle.com/simakov))

* **Papers**:
    - Anton Vakhrushev, Alexander Ryzhkov, Dmitry Simakov, Rinchin Damdinov, Maxim Savchenko, Alexander Tuzhilin ["LightAutoML: AutoML Solution for a Large Financial Services Ecosystem"](https://arxiv.org/pdf/2109.01528.pdf). arXiv:2109.01528, 2021.

* **Articles about LightAutoML**:
    - (English) [LightAutoML vs Titanic: 80% accuracy in several lines of code (Medium)](https://alexmryzhkov.medium.com/lightautoml-preset-usage-tutorial-2cce7da6f936)
    - (English) [Hands-On Python Guide to LightAutoML – An Automatic ML Model Creation Framework (Analytic Indian Mag)](https://analyticsindiamag.com/hands-on-python-guide-to-lama-an-automatic-ml-model-creation-framework/?fbclid=IwAR0f0cVgQWaLI60m1IHMD6VZfmKce0ZXxw-O8VRTdRALsKtty8a-ouJex7g)

[Back to top](#toc)

<a name="contributing"></a>
# Contributing to LightAutoML
If you are interested in contributing to LightAutoML, please read the [Contributing Guide](.github/CONTRIBUTING.md) to get started.

[Back to top](#toc)

<a name="apache"></a>
# License
This project is licensed under the Apache License, Version 2.0. See [LICENSE](https://github.com/AILab-MLTools/LightAutoML/blob/master/LICENSE) file for more details.

[Back to top](#toc)

<a name="developers"></a>
# For developers

## Installation from source code

First of all you need to install [git](https://git-scm.com/downloads) and [poetry](https://python-poetry.org/docs/#installation).

```bash

# Load LAMA source code
git clone https://github.com/AILab-MLTools/LightAutoML.git

cd LightAutoML/

# !!!Choose only one item!!!

# 1. Global installation: Don't create virtual environment
poetry config virtualenvs.create false --local

# 2. Recommended: Create virtual environment inside your project directory
poetry config virtualenvs.in-project true

# For more information read poetry docs

# Install LAMA
poetry lock
poetry install
```

## Build your own custom pipeline:

```python
import pandas as pd
from sklearn.metrics import f1_score

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

df_train = pd.read_csv('../input/titanic/train.csv')
df_test = pd.read_csv('../input/titanic/test.csv')

# define that machine learning problem is binary classification
task = Task("binary")

reader = PandasToPandasReader(task, cv=N_FOLDS, random_state=RANDOM_STATE)

# create a feature selector
model0 = BoostLGBM(
    default_params={'learning_rate': 0.05, 'num_leaves': 64,
    'seed': 42, 'num_threads': N_THREADS}
)
pipe0 = LGBSimpleFeatures()
mbie = ModelBasedImportanceEstimator()
selector = ImportanceCutoffSelector(pipe0, model0, mbie, cutoff=0)

# build first level pipeline for AutoML
pipe = LGBSimpleFeatures()
# stop after 20 iterations or after 30 seconds
params_tuner1 = OptunaTuner(n_trials=20, timeout=30)
model1 = BoostLGBM(
    default_params={'learning_rate': 0.05, 'num_leaves': 128,
    'seed': 1, 'num_threads': N_THREADS}
)
model2 = BoostLGBM(
    default_params={'learning_rate': 0.025, 'num_leaves': 64,
    'seed': 2, 'num_threads': N_THREADS}
)
pipeline_lvl1 = MLPipeline([
    (model1, params_tuner1),
    model2
], pre_selection=selector, features_pipeline=pipe, post_selection=None)

# build second level pipeline for AutoML
pipe1 = LGBSimpleFeatures()
model = BoostLGBM(
    default_params={'learning_rate': 0.05, 'num_leaves': 64,
    'max_bin': 1024, 'seed': 3, 'num_threads': N_THREADS},
    freeze_defaults=True
)
pipeline_lvl2 = MLPipeline([model], pre_selection=None, features_pipeline=pipe1,
 post_selection=None)

# build AutoML pipeline
automl = AutoML(reader, [
    [pipeline_lvl1],
    [pipeline_lvl2],
], skip_conn=False)

# train AutoML and get predictions
oof_pred = automl.fit_predict(df_train, roles = {'target': 'Survived', 'drop': ['PassengerId']})
test_pred = automl.predict(df_test)

pd.DataFrame({
    'PassengerId':df_test.PassengerId,
    'Survived': (test_pred.data[:, 0] > 0.5)*1
}).to_csv('submit.csv', index = False)
```

[Back to top](#toc)

<a name="support"></a>
# Support and feature requests
Seek prompt advice at [Telegram group](https://t.me/lightautoml).

Open bug reports and feature requests on GitHub [issues](https://github.com/AILab-MLTools/LightAutoML/issues).
