# HypEx: Hypotheses and Experiments for Causal Inference

[![Telegram](https://img.shields.io/badge/chat-on%20Telegram-2ba2d9.svg)](https://t.me/lamamatcher)

## Introduction
HypEx (Hypotheses and Experiments) is an addon for the LightAutoML library, designed to automate the causal inference process in data analysis. It is developed to solve matching tasks in a more effective and efficient way. This addon utilizes the Rubin's Causal Model (RCM) approach, a classic method to match closely resembling pairs, thereby ensuring a fair comparison between groups when estimating the effect of a treatment.

The HypEx addon is designed with a fully automated pipeline to calculate Average Treatment Effect (ATE), Average Treatment Effect on the Treated (ATT), and Average Treatment Effect on the Control (ATC). It provides users with a standard interface to execute these estimations and understand the impact of interventions on different subgroups in the population.

Key features of HypEx include automated feature selection with LightAutoML, matching using Faiss KNN for optimal pair selection, application of various data filters, and result validation.

## Features
- Automated Feature Selection: HypEx uses the LightAutoML feature selection process to identify and use the most relevant features for causal inference.
- [Faiss](https://github.com/facebookresearch/faiss) KNN Matching: The addon leverages the power of Faiss library to perform efficient nearest neighbor searches for matching. This ensures that for each treated instance, a control instance that is closest in characteristics is selected, as per Rubin's Causal Model.
- Data Filters: The addon comes with in-built outlier detection and Spearman filters to ensure that the data being used for matching is of high quality.
- Result Validation: HypEx provides three ways for users to validate the results: random treatment validation, random feature validation, and random subset validation.
- Data Tests: HypEx also includes data testing methods such as the Standard Mean Difference (SMD) test, Kolmogorov-Smirnov (KS) test, Population Stability Index (PSI) test, and Repeats test. These tests provide additional checks to ensure the robustness of the estimated effects.

## Quick Start
You can see the examples of usages this addons [here](https://github.com/sb-ai-lab/LightAutoML/blob/master/examples/tutorials/Tutorial_12_Matching.ipynb)

Conclusion
The HypEx addon for LightAutoML is a powerful tool for any data analyst or researcher interested in causal inference. Its automated features, effective matching technique, and rigorous validation and testing methods make it an essential addition to the toolkit when seeking to understand cause and effect in complex datasets.

