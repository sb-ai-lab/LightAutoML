<img src=docs/imgs/lightautoml_logo_color.png />

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lightautoml)](https://pypi.org/project/lightautoml/)
[![PyPI - Version](https://img.shields.io/pypi/v/lightautoml)](https://pypi.org/project/lightautoml)
![pypi - Downloads](https://img.shields.io/pypi/dm/lightautoml?color=green&label=PyPI%20downloads&logo=pypi&logoColor=green)
[![GitHub License](https://img.shields.io/github/license/sb-ai-lab/LightAutoML)](https://github.com/sb-ai-lab/RePlay/blob/main/LICENSE)
[![Telegram](https://img.shields.io/badge/chat-on%20Telegram-2ba2d9.svg)](https://t.me/lightautoml)
<br>
[![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/sb-ai-lab/lightautoml/CI.yml)](https://github.com/sb-ai-lab/lightautoml/actions/workflows/CI.yml?query=branch%3Amain)
![Poetry-Lock](https://img.shields.io/github/workflow/status/sb-ai-lab/LightAutoML/Poetry%20run/master?label=Poetry-Lock)
![Read the Docs](https://img.shields.io/readthedocs/lightautoml)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

LightAutoML (LAMA) is an AutoML framework which provides automatic model creation for the following tasks:
- binary classification
- multiclass classification
- multilabel classification
- regression

Current version of the package handles datasets that have independent samples in each row. I.e. **each row is an object with its specific features and target**.

**Authors**: [Alexander Ryzhkov](https://kaggle.com/alexryzhkov), [Anton Vakhrushev](https://kaggle.com/btbpanda), [Dmitry Simakov](https://kaggle.com/simakov), Rinchin Damdinov, Vasilii Bunakov, Alexander Kirilin, Pavel Shvets.


<a name="toc"></a>
# Table of Contents

* [Installation](#installation)
* [Documentation](https://lightautoml.readthedocs.io/)
* [Quick tour](#quicktour)
* [Resources](#examples)
* [Advanced features](#advancedfeatures)
* [Support and feature requests](#support)
* [Contributing to LightAutoML](#contributing)
* [License](#license)

**Documentation** of LightAutoML is available [here](https://lightautoml.readthedocs.io/), you can also [generate](https://github.com/AILab-MLTools/LightAutoML/blob/master/.github/CONTRIBUTING.md#building-documentation) it.


<a name="installation"></a>
# Installation
To install LAMA framework on your machine from PyPI:
```bash
# Base functionality:
pip install -U lightautoml

# For partial installation use corresponding option
# Extra dependecies: [nlp, cv, report] or use 'all' to install all dependecies
pip install -U lightautoml[nlp]
```

Additionally, run following commands to enable pdf report generation:

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
### Use ready preset for tabular data
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

### LightAutoML as a framework: build your own custom pipeline

```python
import pandas as pd
from sklearn.metrics import f1_score

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

df_train = pd.read_csv('../input/titanic/train.csv')
df_test = pd.read_csv('../input/titanic/test.csv')
N_THREADS = 4

reader = PandasToPandasReader(Task("binary"), cv=5, random_state=42)

# create a feature selector
selector = ImportanceCutoffSelector(
    LGBSimpleFeatures(), 
    BoostLGBM(
        default_params={'learning_rate': 0.05, 'num_leaves': 64,
        'seed': 42, 'num_threads': N_THREADS}
    ),
    ModelBasedImportanceEstimator(), 
    cutoff=0
)

# build first level pipeline for AutoML
pipeline_lvl1 = MLPipeline([
    # first model with hyperparams tuning
    (
        BoostLGBM(
            default_params={'learning_rate': 0.05, 'num_leaves': 128,
            'seed': 1, 'num_threads': N_THREADS}
        ), 
        OptunaTuner(n_trials=20, timeout=30)
    ),
    # second model without hyperparams tuning
    BoostLGBM(
        default_params={'learning_rate': 0.025, 'num_leaves': 64,
        'seed': 2, 'num_threads': N_THREADS}
    )
], pre_selection=selector, features_pipeline=LGBSimpleFeatures(), post_selection=None)

# build second level pipeline for AutoML
pipeline_lvl2 = MLPipeline(
    [
        BoostLGBM(
            default_params={'learning_rate': 0.05, 'num_leaves': 64,
            'max_bin': 1024, 'seed': 3, 'num_threads': N_THREADS},
            freeze_defaults=True
        )
    ], 
    pre_selection=None, 
    features_pipeline=LGBSimpleFeatures(),
    post_selection=None
)

# build AutoML pipeline
automl = AutoML(reader, [
        [pipeline_lvl1],
        [pipeline_lvl2],
    ],
    skip_conn=False
)

# train AutoML and get predictions
oof_pred = automl.fit_predict(df_train, roles = {'target': 'Survived', 'drop': ['PassengerId']})
test_pred = automl.predict(df_test)

pd.DataFrame({
    'PassengerId':df_test.PassengerId,
    'Survived': (test_pred.data[:, 0] > 0.5)*1
}).to_csv('submit.csv', index = False)
```

LighAutoML framework has a lot of ready-to-use parts and extensive customization options, to learn more check out the [resources](#Resources) section.

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
- [Custom ML pipeline elements inside existing ones](https://www.kaggle.com/simakov/lama-custom-automl-pipeline-example)
- [Tabular Playground Series November 2022 competition solution with Neural Networks](https://www.kaggle.com/code/mikhailkuz/lightautoml-nn-happiness)

### Google Colab tutorials and [other examples](examples/):

- [`Tutorial_1_basics.ipynb`](https://colab.research.google.com/github/AILab-MLTools/LightAutoML/blob/master/examples/tutorials/Tutorial_1_basics.ipynb) - get started with LightAutoML on tabular data.
- [`Tutorial_2_WhiteBox_AutoWoE.ipynb`](https://colab.research.google.com/github/AILab-MLTools/LightAutoML/blob/master/examples/tutorials/Tutorial_2_WhiteBox_AutoWoE.ipynb) - creating interpretable models.
- [`Tutorial_3_sql_data_source.ipynb`](https://colab.research.google.com/github/AILab-MLTools/LightAutoML/blob/master/examples/tutorials/Tutorial_3_sql_data_source.ipynb) - shows how to use LightAutoML presets (both standalone and time utilized variants) for solving ML tasks on tabular data from SQL data base instead of CSV.
- [`Tutorial_4_NLP_Interpretation.ipynb`](https://colab.research.google.com/github/AILab-MLTools/LightAutoML/blob/master/examples/tutorials/Tutorial_4_NLP_Interpretation.ipynb) - example of using TabularNLPAutoML preset, LimeTextExplainer.
- [`Tutorial_5_uplift.ipynb`](https://colab.research.google.com/github/AILab-MLTools/LightAutoML/blob/master/examples/tutorials/Tutorial_5_uplift.ipynb) - shows how to use LightAutoML for a uplift-modeling task.
- [`Tutorial_6_custom_pipeline.ipynb`](https://colab.research.google.com/github/AILab-MLTools/LightAutoML/blob/master/examples/tutorials/Tutorial_6_custom_pipeline.ipynb) - shows how to create your own pipeline from specified blocks: pipelines for feature generation and feature selection, ML algorithms, hyperparameter optimization etc.
- [`Tutorial_7_ICE_and_PDP_interpretation.ipynb`](https://colab.research.google.com/github/AILab-MLTools/LightAutoML/blob/master/examples/tutorials/Tutorial_7_ICE_and_PDP_interpretation.ipynb) - shows how to obtain local and global interpretation of model results using ICE and PDP approaches.
- [`Tutorial_8_CV_preset.ipynb`](https://colab.research.google.com/github/AILab-MLTools/LightAutoML/blob/master/examples/tutorials/Tutorial_8_CV_preset.ipynb) - example of using TabularCVAutoML preset in CV multi-class classification task.
- [`Tutorial_9_neural_networks.ipynb`](https://colab.research.google.com/github/AILab-MLTools/LightAutoML/blob/master/examples/tutorials/Tutorial_9_neural_networks.ipynb) - example of using Tabular preset with neural networks.
- [`Tutorial_10_relational_data_with_star_scheme.ipynb`](https://colab.research.google.com/github/AILab-MLTools/LightAutoML/blob/master/examples/tutorials/Tutorial_10_relational_data_with_star_scheme.ipynb) - example of using Tabular preset with neural networks.
- [`Tutorial_11_time_series.ipynb`](https://colab.research.google.com/github/AILab-MLTools/LightAutoML/blob/master/examples/tutorials/Tutorial_11_time_series.ipynb) - example of using Tabular preset with timeseries data.
- [`Tutorial_12_Matching.ipynb`](https://colab.research.google.com/github/AILab-MLTools/LightAutoML/blob/master/examples/tutorials/Tutorial_12_Matching.ipynb) - example of using addon for matchig.


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

<a name="advancedfeatures"></a>
# Advanced features
### GPU and Spark pipelines
Full GPU and Spark pipelines for LightAutoML currently available for developers testing (still in progress). The code and tutorials for:
- GPU pipeline is [available here](https://github.com/Rishat-skoltech/LightAutoML_GPU)
- Spark pipeline is [available here](https://github.com/sb-ai-lab/SLAMA)

<a name="contributing"></a>
# Contributing to LightAutoML
If you are interested in contributing to LightAutoML, please read the [Contributing Guide](.github/CONTRIBUTING.md) to get started.

<a name="support"></a>
# Support and feature requests
Seek prompt advice at [Telegram group](https://t.me/lightautoml).

Open bug reports and feature requests on GitHub [issues](https://github.com/AILab-MLTools/LightAutoML/issues).

<a name="license"></a>
# License
This project is licensed under the Apache License, Version 2.0. See [LICENSE](https://github.com/AILab-MLTools/LightAutoML/blob/master/LICENSE) file for more details.

[Back to top](#toc)
