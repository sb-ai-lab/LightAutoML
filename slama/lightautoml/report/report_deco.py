"""Classes for report generation and add-ons."""

import logging
import os
import warnings

from copy import copy
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from jinja2 import Environment
from jinja2 import FileSystemLoader
from json2html import json2html
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import explained_variance_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
from sklearn.metrics import r2_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from lightautoml.addons.uplift import metrics as uplift_metrics
from lightautoml.addons.uplift.metalearners import TLearner
from lightautoml.addons.uplift.metalearners import XLearner
from lightautoml.addons.uplift.utils import _get_treatment_role


logger = logging.getLogger(__name__)

base_dir = os.path.dirname(__file__)


def extract_params(input_struct):
    params = dict()
    iterator = input_struct if isinstance(input_struct, dict) else input_struct.__dict__
    for key in iterator:
        if key.startswith(("_", "autonlp_params")):
            continue
        value = iterator[key]
        if type(value) in [bool, int, float, str]:
            params[key] = value
        elif value is None:
            params[key] = None
        elif hasattr(value, "__dict__") or isinstance(value, dict):
            params[key] = extract_params(value)
        else:
            params[key] = str(type(value))
    return params


def plot_roc_curve_image(data, path):
    sns.set(style="whitegrid", font_scale=1.5)
    plt.figure(figsize=(10, 10))

    fpr, tpr, _ = roc_curve(data["y_true"], data["y_pred"])
    auc_score = roc_auc_score(data["y_true"], data["y_pred"])

    lw = 2
    plt.plot(fpr, tpr, color="blue", lw=lw, label="Trained model")
    plt.plot([0, 1], [0, 1], color="red", lw=lw, linestyle="--", label="Random model")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    lgd = plt.legend(bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=2)
    plt.xticks(np.arange(0, 1.01, 0.05), rotation=45)
    plt.yticks(np.arange(0, 1.01, 0.05))
    plt.grid(color="gray", linestyle="-", linewidth=1)
    plt.title("ROC curve (GINI = {:.3f})".format(2 * auc_score - 1))
    plt.savefig(path, bbox_extra_artists=(lgd,), bbox_inches="tight")
    plt.close()
    return auc_score


def plot_pr_curve_image(data, path):
    sns.set(style="whitegrid", font_scale=1.5)
    plt.figure(figsize=(10, 10))

    precision, recall, _ = precision_recall_curve(data["y_true"], data["y_pred"])
    ap_score = average_precision_score(data["y_true"], data["y_pred"])

    lw = 2
    plt.plot(recall, precision, color="blue", lw=lw, label="Trained model")
    positive_rate = np.sum(data["y_true"] == 1) / data.shape[0]
    plt.plot(
        [0, 1],
        [positive_rate, positive_rate],
        color="red",
        lw=lw,
        linestyle="--",
        label="Random model",
    )
    plt.xlim([-0.05, 1.05])
    plt.ylim([0.45, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    lgd = plt.legend(bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=2)
    plt.xticks(np.arange(0, 1.01, 0.05), rotation=45)
    plt.yticks(np.arange(0, 1.01, 0.05))
    plt.grid(color="gray", linestyle="-", linewidth=1)
    plt.title("PR curve (AP = {:.3f})".format(ap_score))
    plt.savefig(path, bbox_extra_artists=(lgd,), bbox_inches="tight")
    plt.close()


def plot_preds_distribution_by_bins(data, path):
    sns.set(style="whitegrid", font_scale=1.5)
    fig, axs = plt.subplots(figsize=(16, 10))

    box_plot_data = []
    labels = []
    for name, group in data.groupby("bin"):
        labels.append(name)
        box_plot_data.append(group["y_pred"].values)

    box = axs.boxplot(box_plot_data, patch_artist=True, labels=labels)
    for patch in box["boxes"]:
        patch.set_facecolor("green")
    axs.set_yscale("log")
    axs.set_xlabel("Bin number")
    axs.set_ylabel("Prediction")
    axs.set_title("Distribution of object predictions by bin")

    fig.savefig(path, bbox_inches="tight")
    plt.close()


def plot_distribution_of_logits(data, path):
    sns.set(style="whitegrid", font_scale=1.5)
    fig, axs = plt.subplots(figsize=(16, 10))

    data["proba_logit"] = np.log(data["y_pred"].values / (1 - data["y_pred"].values))
    sns.kdeplot(
        data[data["y_true"] == 0]["proba_logit"],
        shade=True,
        color="r",
        label="Class 0 logits",
        ax=axs,
    )
    sns.kdeplot(
        data[data["y_true"] == 1]["proba_logit"],
        shade=True,
        color="g",
        label="Class 1 logits",
        ax=axs,
    )
    axs.set_xlabel("Logits")
    axs.set_ylabel("Density")
    axs.set_title("Logits distribution of object predictions (by classes)")
    fig.savefig(path, bbox_inches="tight")
    plt.close()


def plot_pie_f1_metric(data, F1_thresh, path):
    tn, fp, fn, tp = confusion_matrix(data["y_true"], (data["y_pred"] > F1_thresh).astype(int)).ravel()
    (_, prec), (_, rec), (_, F1), (_, _) = precision_recall_fscore_support(
        data["y_true"], (data["y_pred"] > F1_thresh).astype(int)
    )

    sns.set(style="whitegrid", font_scale=1.5)
    fig, ax = plt.subplots(figsize=(20, 10), subplot_kw=dict(aspect="equal"))

    recipe = [
        "{} True Positives".format(tp),
        "{} False Positives".format(fp),
        "{} False Negatives".format(fn),
        "{} True Negatives".format(tn),
    ]

    wedges, texts = ax.pie([tp, fp, fn, tn], wedgeprops=dict(width=0.5), startangle=-40)

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(
        arrowprops=dict(arrowstyle="-", color="k"),
        bbox=bbox_props,
        zorder=0,
        va="center",
    )

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2.0 + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(
            recipe[i], xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y), horizontalalignment=horizontalalignment, **kw
        )

    ax.set_title(
        "Trained model: Precision = {:.2f}%, Recall = {:.2f}%, F1-Score = {:.2f}%".format(
            prec * 100, rec * 100, F1 * 100
        )
    )
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return prec, rec, F1


def f1_score_w_co(input_data, min_co=0.01, max_co=0.99, step=0.01):
    data = input_data.copy()
    data["y_pred"] = np.clip(np.ceil(data["y_pred"].values / step) * step, min_co, max_co)

    pos = data["y_true"].sum()
    neg = data["y_true"].shape[0] - pos

    grp = pd.DataFrame(data).groupby("y_pred")["y_true"].agg(["sum", "count"])
    grp.sort_index(inplace=True)

    grp["fp"] = grp["sum"].cumsum()
    grp["tp"] = pos - grp["fp"]
    grp["tn"] = (grp["count"] - grp["sum"]).cumsum()
    grp["fn"] = neg - grp["tn"]

    grp["pr"] = grp["tp"] / (grp["tp"] + grp["fp"])
    grp["rec"] = grp["tp"] / (grp["tp"] + grp["fn"])

    grp["f1_score"] = 2 * (grp["pr"] * grp["rec"]) / (grp["pr"] + grp["rec"])

    best_score = grp["f1_score"].max()
    best_co = grp.index.values[grp["f1_score"] == best_score].mean()

    # print((y_pred < best_co).mean())

    return best_score, best_co


def get_bins_table(data):
    bins_table = data.groupby("bin").agg({"y_true": [len, np.mean], "y_pred": [np.min, np.mean, np.max]}).reset_index()
    bins_table.columns = [
        "Bin number",
        "Amount of objects",
        "Mean target",
        "Min probability",
        "Average probability",
        "Max probability",
    ]
    return bins_table.to_html(index=False)


# Regression plots:


def plot_target_distribution_1(data, path):
    sns.set(style="whitegrid", font_scale=1.5)
    fig, axs = plt.subplots(2, 1, figsize=(16, 20))

    sns.kdeplot(data["y_true"], shade=True, color="g", ax=axs[0])
    axs[0].set_xlabel("Target value")
    axs[0].set_ylabel("Density")
    axs[0].set_title("Target distribution (y_true)")

    sns.kdeplot(data["y_pred"], shade=True, color="r", ax=axs[1])
    axs[1].set_xlabel("Target value")
    axs[1].set_ylabel("Density")
    axs[1].set_title("Target distribution (y_pred)")

    fig.savefig(path, bbox_inches="tight")
    plt.close()


def plot_target_distribution_2(data, path):
    sns.set(style="whitegrid", font_scale=1.5)
    fig, axs = plt.subplots(figsize=(16, 10))

    sns.kdeplot(data["y_true"], shade=True, color="g", label="y_true", ax=axs)
    sns.kdeplot(data["y_pred"], shade=True, color="r", label="y_pred", ax=axs)
    axs.set_xlabel("Target value")
    axs.set_ylabel("Density")
    axs.set_title("Target distribution")

    fig.savefig(path, bbox_inches="tight")
    plt.close()


def plot_target_distribution(data, path):
    data_pred = pd.DataFrame({"Target value": data["y_pred"]})
    data_pred["source"] = "y_pred"
    data_true = pd.DataFrame({"Target value": data["y_true"]})
    data_true["source"] = "y_true"
    data = pd.concat([data_pred, data_true], ignore_index=True)

    sns.set(style="whitegrid", font_scale=1.5)
    g = sns.displot(
        data,
        x="Target value",
        row="source",
        height=9,
        aspect=1.5,
        kde=True,
        color="m",
        facet_kws=dict(margin_titles=True),
    )
    g.fig.suptitle("Target distribution")
    g.fig.tight_layout()
    g.fig.subplots_adjust(top=0.95)

    g.fig.savefig(path, bbox_inches="tight")
    plt.close()


def plot_error_hist(data, path):
    sns.set(style="whitegrid", font_scale=1.5)
    fig, ax = plt.subplots(figsize=(16, 10))

    sns.kdeplot(data["y_pred"] - data["y_true"], shade=True, color="m", ax=ax)
    ax.set_xlabel("Error = y_pred - y_true")
    ax.set_ylabel("Density")
    ax.set_title("Error histogram")

    fig.savefig(path, bbox_inches="tight")
    plt.close()


def plot_reg_scatter(data, path):
    sns.set(style="whitegrid", font_scale=1.5)
    g = sns.jointplot(
        x="y_pred",
        y="y_true",
        data=data,
        kind="reg",
        truncate=False,
        color="m",
        height=14,
    )
    g.fig.suptitle("Scatter plot")
    g.fig.tight_layout()
    g.fig.subplots_adjust(top=0.95)

    g.fig.savefig(path, bbox_inches="tight")
    plt.close()


# Multiclass plots:


def plot_confusion_matrix(data, path):
    sns.set(style="whitegrid", font_scale=1.5)
    fig, ax = plt.subplots(figsize=(16, 12))

    cmat = confusion_matrix(data["y_true"], data["y_pred"], normalize="true")
    sns.heatmap(cmat, annot=True, linewidths=0.5, cmap="Purples", ax=ax)
    ax.set_xlabel("y_pred")
    ax.set_ylabel("y_true")
    ax.set_title("Confusion matrix")

    fig.savefig(path, bbox_inches="tight")
    plt.close()


# Feature importance


def plot_feature_importance(feat_imp, path, features_max=100):
    sns.set(style="whitegrid", font_scale=1.5)
    fig, axs = plt.subplots(figsize=(16, features_max / 2.5))
    sns.barplot(x="Importance", y="Feature", data=feat_imp[:features_max], ax=axs, color="m")
    plt.savefig(path, bbox_inches="tight")
    plt.close()


class ReportDeco:
    """
    Decorator to wrap :class:`~lightautoml.automl.base.AutoML` class to generate html report on ``fit_predict`` and ``predict``.

    Example:

        >>> report_automl = ReportDeco(output_path="output_path", report_file_name="report_file_name")(automl).
        >>> report_automl.fit_predict(train_data)
        >>> report_automl.predict(test_data)

    Report will be generated at output_path/report_file_name automatically.

    Warning:
         Do not use it just to inference (if you don"t need report), because:

            - It needs target variable to calc performance metrics.
            - It takes additional time to generate report.
            - Dump of decorated automl takes more memory to store.

    To get unwrapped fitted instance to pickle
    and inferecne access ``report_automl.model`` attribute.

    """

    @property
    def model(self):
        """Get unwrapped model.

        Returns:
            model.

        """
        return self._model

    @property
    def mapping(self):
        return self._model.reader.class_mapping

    @property
    def task(self):
        return self._model.reader.task._name

    def __init__(self, *args, **kwargs):
        """

        Note:
            Valid kwargs are:

                - output_path: Folder with report files.
                - report_file_name: Name of main report file.

        Args:
            *args: Arguments.
            **kwargs: Additional parameters.

        """
        if not kwargs:
            kwargs = {}

        # default params
        self.fi_params = {"method": "fast", "n_sample": 100_000}
        self.interpretation_params = {
            "top_n_features": 5,
            "top_n_categories": 10,
            "ton_n_classes": 10,
            "n_bins": 30,
            "datetime_level": "year",
            "n_sample": 100_000,
        }

        fi_input_params = kwargs.get("fi_params", {})
        self.fi_params.update(fi_input_params)
        interpretation_input_params = kwargs.get("interpretation_params", {})
        self.interpretation_params.update(interpretation_input_params)
        self.interpretation = kwargs.get("interpretation", False)

        self.n_bins = kwargs.get("n_bins", 20)
        self.template_path = kwargs.get("template_path", os.path.join(base_dir, "lama_report_templates/"))
        self.output_path = kwargs.get("output_path", "lama_report/")
        self.report_file_name = kwargs.get("report_file_name", "lama_interactive_report.html")
        self.pdf_file_name = kwargs.get("pdf_file_name", None)

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)

        self._base_template_path = "lama_base_template.html"
        self._model_section_path = "model_section.html"
        self._train_set_section_path = "train_set_section.html"
        self._results_section_path = "results_section.html"
        self._fi_section_path = "feature_importance_section.html"
        self._interpretation_section_path = "interpretation_section.html"
        self._interpretation_subsection_path = "interpretation_subsection.html"

        self._inference_section_path = {
            "binary": "binary_inference_section.html",
            "reg": "reg_inference_section.html",
            "multiclass": "multiclass_inference_section.html",
        }

        self.title = "LAMA report"
        if self.interpretation:
            self.sections_order = [
                "intro",
                "model",
                "train_set",
                "fi",
                "interpretation",
                "results",
            ]
            self._interpretation_top = []
        else:
            self.sections_order = ["intro", "model", "train_set", "fi", "results"]
        self._sections = {}
        self._sections["intro"] = "<p>This report was generated automatically.</p>"
        self._model_results = []

        self.generate_report()

    def __call__(self, model):
        self._model = model

        # add informataion to report
        self._model_name = model.__class__.__name__
        self._model_parameters = json2html.convert(extract_params(model))
        self._model_summary = None

        self._sections = {}
        self._sections["intro"] = "<p>This report was generated automatically.</p>"
        self._model_results = []
        self._n_test_sample = 0

        self._generate_model_section()
        self.generate_report()
        return self

    def _binary_classification_details(self, data):
        self._inference_content["sample_bins_table"] = get_bins_table(data)
        prec, rec, F1 = plot_pie_f1_metric(
            data,
            self._F1_thresh,
            path=os.path.join(self.output_path, self._inference_content["pie_f1_metric"]),
        )
        auc_score = plot_roc_curve_image(
            data,
            path=os.path.join(self.output_path, self._inference_content["roc_curve"]),
        )
        plot_pr_curve_image(
            data,
            path=os.path.join(self.output_path, self._inference_content["pr_curve"]),
        )
        plot_preds_distribution_by_bins(
            data,
            path=os.path.join(self.output_path, self._inference_content["preds_distribution_by_bins"]),
        )
        plot_distribution_of_logits(
            data,
            path=os.path.join(self.output_path, self._inference_content["distribution_of_logits"]),
        )
        return auc_score, prec, rec, F1

    def _regression_details(self, data):
        # graphics
        plot_target_distribution(
            data,
            path=os.path.join(self.output_path, self._inference_content["target_distribution"]),
        )
        plot_error_hist(
            data,
            path=os.path.join(self.output_path, self._inference_content["error_hist"]),
        )
        plot_reg_scatter(
            data,
            path=os.path.join(self.output_path, self._inference_content["scatter_plot"]),
        )
        # metrics
        mean_ae = mean_absolute_error(data["y_true"], data["y_pred"])
        median_ae = median_absolute_error(data["y_true"], data["y_pred"])
        mse = mean_squared_error(data["y_true"], data["y_pred"])
        r2 = r2_score(data["y_true"], data["y_pred"])
        evs = explained_variance_score(data["y_true"], data["y_pred"])
        return mean_ae, median_ae, mse, r2, evs

    def _multiclass_details(self, data):
        y_true = data["y_true"]
        y_pred = data["y_pred"]
        # precision
        p_micro = precision_score(y_true, y_pred, average="micro")
        p_macro = precision_score(y_true, y_pred, average="macro")
        p_weighted = precision_score(y_true, y_pred, average="weighted")
        # recall
        r_micro = recall_score(y_true, y_pred, average="micro")
        r_macro = recall_score(y_true, y_pred, average="macro")
        r_weighted = recall_score(y_true, y_pred, average="weighted")
        # f1-score
        f_micro = f1_score(y_true, y_pred, average="micro")
        f_macro = f1_score(y_true, y_pred, average="macro")
        f_weighted = f1_score(y_true, y_pred, average="weighted")

        # classification report for features
        if self.mapping:
            classes = sorted(self.mapping, key=self.mapping.get)
        else:
            classes = np.arange(self._N_classes)
        p, r, f, s = precision_recall_fscore_support(y_true, y_pred)
        cls_report = pd.DataFrame(
            {
                "Class name": classes,
                "Precision": p,
                "Recall": r,
                "F1-score": f,
                "Support": s,
            }
        )
        self._inference_content["classification_report"] = cls_report.to_html(
            index=False, float_format="{:.4f}".format, justify="left"
        )

        plot_confusion_matrix(
            data,
            path=os.path.join(self.output_path, self._inference_content["confusion_matrix"]),
        )

        return [
            p_micro,
            p_macro,
            p_weighted,
            r_micro,
            r_macro,
            r_weighted,
            f_micro,
            f_macro,
            f_weighted,
        ]

    def _collect_data(self, preds, sample):
        data = pd.DataFrame({"y_true": sample[self._target].values})
        if self.task in "multiclass":
            if self.mapping is not None:
                data["y_true"] = np.array([self.mapping[y] for y in data["y_true"].values])
            data["y_pred"] = preds._data.argmax(axis=1)
            data = data[~np.isnan(preds._data).any(axis=1)]
        else:
            data["y_pred"] = preds._data[:, 0]
            data.sort_values("y_pred", ascending=False, inplace=True)
            data["bin"] = (np.arange(data.shape[0]) / data.shape[0] * self.n_bins).astype(int)
            data = data[~data["y_pred"].isnull()]
        return data

    def fit_predict(self, *args, **kwargs):
        """Wrapped ``automl.fit_predict`` method.

        Valid args, kwargs are the same as wrapped automl.

        Args:
            *args: Arguments.
            **kwargs: Additional parameters.

        Returns:
            OOF predictions.

        """
        # TODO: parameters parsing in general case

        preds = self._model.fit_predict(*args, **kwargs)
        train_data = kwargs["train_data"] if "train_data" in kwargs else args[0]
        input_roles = kwargs["roles"] if "roles" in kwargs else args[1]
        self._target = input_roles["target"]
        valid_data = kwargs.get("valid_data", None)
        if valid_data is None:
            data = self._collect_data(preds, train_data)
        else:
            data = self._collect_data(preds, valid_data)
        self._inference_content = {}
        if self.task == "binary":
            # filling for html
            self._inference_content = {}
            self._inference_content["roc_curve"] = "valid_roc_curve.png"
            self._inference_content["pr_curve"] = "valid_pr_curve.png"
            self._inference_content["pie_f1_metric"] = "valid_pie_f1_metric.png"
            self._inference_content["preds_distribution_by_bins"] = "valid_preds_distribution_by_bins.png"
            self._inference_content["distribution_of_logits"] = "valid_distribution_of_logits.png"
            # graphics and metrics
            _, self._F1_thresh = f1_score_w_co(data)
            auc_score, prec, rec, F1 = self._binary_classification_details(data)
            # update model section
            evaluation_parameters = ["AUC-score", "Precision", "Recall", "F1-score"]
            self._model_summary = pd.DataFrame(
                {
                    "Evaluation parameter": evaluation_parameters,
                    "Validation sample": [auc_score, prec, rec, F1],
                }
            )
        elif self.task == "reg":
            # filling for html
            self._inference_content["target_distribution"] = "valid_target_distribution.png"
            self._inference_content["error_hist"] = "valid_error_hist.png"
            self._inference_content["scatter_plot"] = "valid_scatter_plot.png"
            # graphics and metrics
            mean_ae, median_ae, mse, r2, evs = self._regression_details(data)
            # model section
            evaluation_parameters = [
                "Mean absolute error",
                "Median absolute error",
                "Mean squared error",
                "R^2 (coefficient of determination)",
                "Explained variance",
            ]
            self._model_summary = pd.DataFrame(
                {
                    "Evaluation parameter": evaluation_parameters,
                    "Validation sample": [mean_ae, median_ae, mse, r2, evs],
                }
            )
        elif self.task == "multiclass":
            self._N_classes = len(train_data[self._target].drop_duplicates())
            self._inference_content["confusion_matrix"] = "valid_confusion_matrix.png"

            index_names = np.array([["Precision", "Recall", "F1-score"], ["micro", "macro", "weighted"]])
            index = pd.MultiIndex.from_product(index_names, names=["Evaluation metric", "Average"])

            summary = self._multiclass_details(data)
            self._model_summary = pd.DataFrame({"Validation sample": summary}, index=index)

        self._inference_content["title"] = "Results on validation sample"

        self._generate_model_section()

        # generate train data section
        self._train_data_overview = self._data_genenal_info(train_data)
        self._describe_roles(train_data)
        self._describe_dropped_features(train_data)
        self._generate_train_set_section()
        # generate fit_predict section
        self._generate_inference_section()
        # generate feature importance and interpretation sections
        self._generate_fi_section(valid_data)
        if self.interpretation:
            self._generate_interpretation_section(valid_data)

        self.generate_report()
        return preds

    def predict(self, *args, **kwargs):
        """Wrapped automl.predict method.

        Valid args, kwargs are the same as wrapped automl.

        Args:
            *args: arguments.
            **kwargs: additional parameters.

        Returns:
            predictions.

        """
        self._n_test_sample += 1
        # get predictions
        test_preds = self._model.predict(*args, **kwargs)

        test_data = kwargs["test"] if "test" in kwargs else args[0]
        data = self._collect_data(test_preds, test_data)

        if self.task == "binary":
            # filling for html
            self._inference_content = {}
            self._inference_content["roc_curve"] = "test_roc_curve_{}.png".format(self._n_test_sample)
            self._inference_content["pr_curve"] = "test_pr_curve_{}.png".format(self._n_test_sample)
            self._inference_content["pie_f1_metric"] = "test_pie_f1_metric_{}.png".format(self._n_test_sample)
            self._inference_content["bins_preds"] = "test_bins_preds_{}.png".format(self._n_test_sample)
            self._inference_content["preds_distribution_by_bins"] = "test_preds_distribution_by_bins_{}.png".format(
                self._n_test_sample
            )
            self._inference_content["distribution_of_logits"] = "test_distribution_of_logits_{}.png".format(
                self._n_test_sample
            )
            # graphics and metrics
            auc_score, prec, rec, F1 = self._binary_classification_details(data)

            if self._n_test_sample >= 2:
                self._model_summary["Test sample {}".format(self._n_test_sample)] = [
                    auc_score,
                    prec,
                    rec,
                    F1,
                ]
            else:
                self._model_summary["Test sample"] = [auc_score, prec, rec, F1]

        elif self.task == "reg":
            # filling for html
            self._inference_content = {}
            self._inference_content["target_distribution"] = "test_target_distribution_{}.png".format(
                self._n_test_sample
            )
            self._inference_content["error_hist"] = "test_error_hist_{}.png".format(self._n_test_sample)
            self._inference_content["scatter_plot"] = "test_scatter_plot_{}.png".format(self._n_test_sample)
            # graphics
            mean_ae, median_ae, mse, r2, evs = self._regression_details(data)
            # update model section
            if self._n_test_sample >= 2:
                self._model_summary["Test sample {}".format(self._n_test_sample)] = [
                    mean_ae,
                    median_ae,
                    mse,
                    r2,
                    evs,
                ]
            else:
                self._model_summary["Test sample"] = [mean_ae, median_ae, mse, r2, evs]

        elif self.task == "multiclass":
            self._inference_content["confusion_matrix"] = "test_confusion_matrix_{}.png".format(self._n_test_sample)
            test_summary = self._multiclass_details(data)
            if self._n_test_sample >= 2:
                self._model_summary["Test sample {}".format(self._n_test_sample)] = test_summary
            else:
                self._model_summary["Test sample"] = test_summary

        # layout depends on number of test samples
        if self._n_test_sample >= 2:
            self._inference_content["title"] = "Results on test sample {}".format(self._n_test_sample)

        else:
            self._inference_content["title"] = "Results on test sample"

        # update model section
        self._generate_model_section()
        # generate predict section
        self._generate_inference_section()

        self.generate_report()
        return test_preds

    def _generate_fi_section(self, valid_data):
        if (
            self.fi_params["method"] == "accurate"
            and valid_data is not None
            and valid_data.shape[0] > self.fi_params["n_sample"]
        ):
            valid_data = valid_data.sample(n=self.fi_params["n_sample"])
            print(
                "valid_data was sampled for feature importance calculation: n_sample = {}".format(
                    self.fi_params["n_sample"]
                )
            )

        if self.fi_params["method"] == "accurate" and valid_data is None:
            # raise ValueError("You must set valid_data with accurate feature importance method")
            self.fi_params["method"] = "fast"
            warnings.warn(
                "You must set valid_data with 'accurate' feature importance method. Changed to 'fast' automatically."
            )

        self.feat_imp = self._model.get_feature_scores(
            calc_method=self.fi_params["method"], data=valid_data, silent=False
        )
        if self.feat_imp is None:
            fi_path = None
        else:
            fi_path = "feature_importance.png"
            plot_feature_importance(self.feat_imp, path=os.path.join(self.output_path, fi_path))
        # add to _sections
        fi_content = {
            "fi_method": self.fi_params["method"],
            "feature_importance": fi_path,
        }
        env = Environment(loader=FileSystemLoader(searchpath=self.template_path))
        fi_section = env.get_template(self._fi_section_path).render(fi_content)
        self._sections["fi"] = fi_section

    def _generate_interpretation_content(self, test_data):
        self._interpretation_content = {}
        if test_data is None:
            self._interpretation_content["interpretation_top"] = None
            return
        if self.feat_imp is None:
            interpretation_feat_list = list(self._model.reader._roles.keys())[
                : self.interpretation_params["top_n_features"]
            ]
        else:
            interpretation_feat_list = self.feat_imp["Feature"].values[: self.interpretation_params["top_n_features"]]
        for feature_name in interpretation_feat_list:
            interpretaton_subsection = {}
            interpretaton_subsection["feature_name"] = feature_name
            interpretaton_subsection["feature_interpretation_plot"] = feature_name + "_interpretation.png"
            self._plot_pdp(
                test_data,
                feature_name,
                path=os.path.join(
                    self.output_path,
                    interpretaton_subsection["feature_interpretation_plot"],
                ),
            )
            env = Environment(loader=FileSystemLoader(searchpath=self.template_path))
            interpretation_subsection = env.get_template(self._interpretation_subsection_path).render(
                interpretaton_subsection
            )
            self._interpretation_top.append(interpretation_subsection)
            print(f"Interpretation info for {feature_name} appended")
        self._interpretation_content["interpretation_top"] = self._interpretation_top

    def _generate_interpretation_section(self, test_data):
        if test_data is not None and test_data.shape[0] > self.interpretation_params["n_sample"]:
            test_data = test_data.sample(n=self.interpretation_params["n_sample"])
        self._generate_interpretation_content(test_data)
        env = Environment(loader=FileSystemLoader(searchpath=self.template_path))
        interpretation_section = env.get_template(self._interpretation_section_path).render(
            self._interpretation_content
        )
        self._sections["interpretation"] = interpretation_section

    def _plot_pdp(self, test_data, feature_name, path):
        feature_role = self._model.reader._roles[feature_name].name
        # I. Count interpretation
        print("Calculating interpretation for {}:".format(feature_name))
        grid, ys, counts = self._model.get_individual_pdp(
            test_data=test_data,
            feature_name=feature_name,
            n_bins=self.interpretation_params["n_bins"],
            top_n_categories=self.interpretation_params["top_n_categories"],
            datetime_level=self.interpretation_params["datetime_level"],
        )
        # II. Plot pdp
        sns.set(style="whitegrid", font_scale=1.5)
        fig, axs = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={"height_ratios": [3, 1]})
        axs[0].set_title("PDP plot: " + feature_name)
        n_classes = ys[0].shape[1]
        if n_classes == 1:
            data = pd.concat(
                [pd.DataFrame({"x": grid[i], "y": ys[i].ravel()}) for i, _ in enumerate(grid)]
            ).reset_index(drop=True)
            if feature_role in ["Numeric", "Datetime"]:
                g0 = sns.lineplot(data=data, x="x", y="y", ax=axs[0], color="m")
            else:
                g0 = sns.boxplot(data=data, x="x", y="y", ax=axs[0], showfliers=False, color="m")
        else:
            if self.mapping:
                classes = sorted(self.mapping, key=self.mapping.get)[: self.interpretation_params["top_n_classes"]]
            else:
                classes = np.arange(min(n_classes, self.interpretation_params["top_n_classes"]))
            data = pd.concat(
                [
                    pd.DataFrame({"x": grid[i], "y": ys[i][:, k], "class": name})
                    for i, _ in enumerate(grid)
                    for k, name in enumerate(classes)
                ]
            ).reset_index(drop=True)
            if self._model.reader._roles[feature_name].name in ["Numeric", "Datetime"]:
                g0 = sns.lineplot(data=data, x="x", y="y", hue="class", ax=axs[0])
            else:
                g0 = sns.boxplot(data=data, x="x", y="y", hue="class", ax=axs[0], showfliers=False)
        g0.set(ylabel="y_pred")
        # III. Plot distribution
        counts = np.array(counts) / sum(counts)
        if feature_role == "Numeric":
            g0.set(xlabel="feature value")
            g1 = sns.histplot(test_data[feature_name], kde=True, color="gray", ax=axs[1])
        elif feature_role == "Category":
            g0.set(xlabel=None)
            axs[0].set_xticklabels(grid, rotation=90)
            g1 = sns.barplot(x=grid, y=counts, ax=axs[1], color="gray")
        else:
            g0.set(xlabel=self.interpretation_params["datetime_level"])
            g1 = sns.barplot(x=grid, y=counts, ax=axs[1], color="gray")
        g1.set(xlabel=None)
        g1.set(ylabel="Frequency")
        g1.set(xticklabels=[])
        # IV. Save picture
        plt.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close()

    def _data_genenal_info(self, data):
        general_info = pd.DataFrame(columns=["Parameter", "Value"])
        general_info.loc[0] = ("Number of records", data.shape[0])
        general_info.loc[1] = ("Total number of features", data.shape[1])
        general_info.loc[2] = ("Used features", len(self._model.reader._used_features))
        general_info.loc[3] = (
            "Dropped features",
            len(self._model.reader._dropped_features),
        )
        # general_info.loc[4] = ("Number of positive cases", np.sum(data[self._target] == 1))
        # general_info.loc[5] = ("Number of negative cases", np.sum(data[self._target] == 0))
        return general_info.to_html(index=False, justify="left")

    def _describe_roles(self, train_data):

        # detect feature roles
        roles = self._model.reader._roles
        numerical_features = [feat_name for feat_name in roles if roles[feat_name].name == "Numeric"]
        categorical_features = [feat_name for feat_name in roles if roles[feat_name].name == "Category"]
        datetime_features = [feat_name for feat_name in roles if roles[feat_name].name == "Datetime"]

        # numerical roles
        numerical_features_df = []
        for feature_name in numerical_features:
            item = {"Feature name": feature_name}
            item["NaN ratio"] = "{:.4f}".format(train_data[feature_name].isna().sum() / train_data.shape[0])
            values = train_data[feature_name].dropna().values
            item["min"] = np.min(values)
            item["quantile_25"] = np.quantile(values, 0.25)
            item["average"] = np.mean(values)
            item["median"] = np.median(values)
            item["quantile_75"] = np.quantile(values, 0.75)
            item["max"] = np.max(values)
            numerical_features_df.append(item)
        if numerical_features_df == []:
            self._numerical_features_table = None
        else:
            self._numerical_features_table = pd.DataFrame(numerical_features_df).to_html(
                index=False, float_format="{:.2f}".format, justify="left"
            )
        # categorical roles
        categorical_features_df = []
        for feature_name in categorical_features:
            item = {"Feature name": feature_name}
            item["NaN ratio"] = "{:.4f}".format(train_data[feature_name].isna().sum() / train_data.shape[0])
            value_counts = train_data[feature_name].value_counts(normalize=True)
            values = value_counts.index.values
            counts = value_counts.values
            item["Number of unique values"] = len(counts)
            item["Most frequent value"] = values[0]
            item["Occurance of most frequent"] = "{:.1f}%".format(100 * counts[0])
            item["Least frequent value"] = values[-1]
            item["Occurance of least frequent"] = "{:.1f}%".format(100 * counts[-1])
            categorical_features_df.append(item)
        if categorical_features_df == []:
            self._categorical_features_table = None
        else:
            self._categorical_features_table = pd.DataFrame(categorical_features_df).to_html(
                index=False, justify="left"
            )
        # datetime roles
        datetime_features_df = []
        for feature_name in datetime_features:
            item = {"Feature name": feature_name}
            item["NaN ratio"] = "{:.4f}".format(train_data[feature_name].isna().sum() / train_data.shape[0])
            values = train_data[feature_name].dropna().values
            item["min"] = np.min(values)
            item["max"] = np.max(values)
            item["base_date"] = self._model.reader._roles[feature_name].base_date
            datetime_features_df.append(item)
        if datetime_features_df == []:
            self._datetime_features_table = None
        else:
            self._datetime_features_table = pd.DataFrame(datetime_features_df).to_html(index=False, justify="left")

    def _describe_dropped_features(self, train_data):
        self._max_nan_rate = self._model.reader.max_nan_rate
        self._max_constant_rate = self._model.reader.max_constant_rate
        self._features_dropped_list = self._model.reader._dropped_features
        # dropped features table
        dropped_list = [col for col in self._features_dropped_list if col != self._target]
        if dropped_list == []:
            self._dropped_features_table = None
        else:
            dropped_nan_ratio = train_data[dropped_list].isna().sum() / train_data.shape[0]
            dropped_most_occured = pd.Series(np.nan, index=dropped_list)
            for col in dropped_list:
                col_most_occured = train_data[col].value_counts(normalize=True).values
                if len(col_most_occured) > 0:
                    dropped_most_occured[col] = col_most_occured[0]
            dropped_features_table = pd.DataFrame(
                {"nan_rate": dropped_nan_ratio, "constant_rate": dropped_most_occured}
            )
            self._dropped_features_table = (
                dropped_features_table.reset_index()
                .rename(columns={"index": "Название переменной"})
                .to_html(index=False, justify="left")
            )

    def _generate_model_section(self):
        model_summary = None
        if self._model_summary is not None:
            model_summary = self._model_summary.to_html(
                index=self.task == "multiclass",
                justify="left",
                float_format="{:.4f}".format,
            )

        env = Environment(loader=FileSystemLoader(searchpath=self.template_path))
        model_section = env.get_template(self._model_section_path).render(
            model_name=self._model_name,
            model_parameters=self._model_parameters,
            model_summary=model_summary,
        )
        self._sections["model"] = model_section

    def _generate_train_set_section(self):
        env = Environment(loader=FileSystemLoader(searchpath=self.template_path))
        train_set_section = env.get_template(self._train_set_section_path).render(
            train_data_overview=self._train_data_overview,
            numerical_features_table=self._numerical_features_table,
            categorical_features_table=self._categorical_features_table,
            datetime_features_table=self._datetime_features_table,
            target=self._target,
            max_nan_rate=self._max_nan_rate,
            max_constant_rate=self._max_constant_rate,
            dropped_features_table=self._dropped_features_table,
        )
        self._sections["train_set"] = train_set_section

    def _generate_inference_section(self):
        env = Environment(loader=FileSystemLoader(searchpath=self.template_path))
        inference_section = env.get_template(self._inference_section_path[self.task]).render(self._inference_content)
        self._model_results.append(inference_section)

    def _generate_results_section(self):
        if self._model_results:
            env = Environment(loader=FileSystemLoader(searchpath=self.template_path))
            results_section = env.get_template(self._results_section_path).render(model_results=self._model_results)
            self._sections["results"] = results_section

    def generate_report(self):
        # collection sections
        self._generate_results_section()
        sections_list = []
        for sec_name in self.sections_order:
            if sec_name in self._sections:
                sections_list.append(self._sections[sec_name])
        # put sections inside
        env = Environment(loader=FileSystemLoader(searchpath=self.template_path))
        report = env.get_template(self._base_template_path).render(
            title=self.title, sections=sections_list, pdf=self.pdf_file_name
        )

        with open(os.path.join(self.output_path, self.report_file_name), "w", encoding="utf-8") as f:
            f.write(report)

        if self.pdf_file_name:
            try:
                from weasyprint import HTML

                HTML(string=report, base_url=self.output_path).write_pdf(
                    os.path.join(self.output_path, self.pdf_file_name)
                )
            except ModuleNotFoundError:
                print("Can't generate PDF report: check manual for installing pdf extras.")


_default_wb_report_params = {
    "automl_date_column": "",
    "report_name": "autowoe_report.html",
    "report_version_id": 1,
    "city": "",
    "model_aim": "",
    "model_name": "",
    "zakazchik": "",
    "high_level_department": "",
    "ds_name": "",
    "target_descr": "",
    "non_target_descr": "",
}


class ReportDecoWhitebox(ReportDeco):
    """
    Special report wrapper for :class:`~lightautoml.automl.presets.whitebox_presets.WhiteBoxPreset`.
    Usage case is the same as main
    :class:`~lightautoml.report.report_deco.ReportDeco` class.
    It generates same report as :class:`~lightautoml.report.report_deco.ReportDeco` ,
    but with additional whitebox report part.

    Difference:

        - report_automl.predict gets additional report argument.
          It stands for updating whitebox report part.
          Calling ``report_automl.predict(test_data, report=True)``
          will update test part of whitebox report.
          Calling ``report_automl.predict(test_data, report=False)``
          will extend general report with.
          New data and keeps whitebox part as is (much more faster).
        - :class:`~lightautoml.automl.presets.whitebox_presets.WhiteBoxPreset`
          should be created with parameter ``general_params={"report": True}``
          to get white box report part.
          If ``general_params`` set to ``{"report": False}``,
          only standard ReportDeco part will be created (much faster).

    """

    @property
    def model(self):
        """Get unwrapped WhiteBox.

        Returns:
            model.

        """
        # this is made to remove heavy whitebox inner report deco
        model = copy(self._model)
        try:
            model_wo_report = model.whitebox.model
        except AttributeError:
            return self._model

        pipe = copy(self._model.levels[0][0])
        ml_algo = copy(pipe.ml_algos[0])

        ml_algo.models = [model_wo_report]
        pipe.ml_algos = [ml_algo]

        model.levels = [[pipe]]

        return model

    @property
    def content(self):
        return self._model.whitebox._ReportDeco__stat

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.wb_report_params = copy(_default_wb_report_params)

        # self.wb_report_params = wb_report_params
        self.wb_report_params["output_path"] = self.output_path
        self._whitebox_section_path = "whitebox_section.html"
        self.sections_order.append("whitebox")

    def fit_predict(self, *args, **kwargs):
        """Wrapped :meth:`AutoML.fit_predict` method.

        Valid args, kwargs are the same as wrapped automl.

        Args:
            *args: Arguments.
            **kwargs: Additional parameters.

        Returns:
            OOF predictions.

        """
        predict_proba = super().fit_predict(*args, **kwargs)

        if self._model.general_params["report"]:
            self._generate_whitebox_section()
        else:
            logger.info2("Whitebox part is not created. Fit WhiteBox with general_params['report'] = True")

        self.generate_report()
        return predict_proba

    def predict(self, *args, **kwargs):
        """Wrapped :meth:`AutoML.predict` method.

        Valid args, kwargs are the same as wrapped automl.

        Args:
            *args: Arguments.
            **kwargs: Additional parameters.

        Returns:
            Predictions.

        """
        if len(args) >= 2:
            args = (args[0],)

        kwargs["report"] = self._model.general_params["report"]

        predict_proba = super().predict(*args, **kwargs)

        if self._model.general_params["report"]:
            self._generate_whitebox_section()
        else:
            logger.info2("Whitebox part is not created. Fit WhiteBox with general_params['report'] = True")

        self.generate_report()
        return predict_proba

    def _generate_whitebox_section(self):
        self._model.whitebox.generate_report(self.wb_report_params)
        content = self.content.copy()

        if self._n_test_sample >= 2:
            content["n_test_sample"] = self._n_test_sample
        content["model_coef"] = pd.DataFrame(content["model_coef"], columns=["Feature name", "Coefficient"]).to_html(
            index=False
        )
        content["p_vals"] = pd.DataFrame(content["p_vals"], columns=["Feature name", "P-value"]).to_html(index=False)
        content["p_vals_test"] = pd.DataFrame(content["p_vals_test"], columns=["Feature name", "P-value"]).to_html(
            index=False
        )
        content["train_vif"] = pd.DataFrame(content["train_vif"], columns=["Feature name", "VIF value"]).to_html(
            index=False
        )
        content["psi_total"] = pd.DataFrame(content["psi_total"], columns=["Feature name", "PSI value"]).to_html(
            index=False
        )
        content["psi_zeros"] = pd.DataFrame(content["psi_zeros"], columns=["Feature name", "PSI value"]).to_html(
            index=False
        )
        content["psi_ones"] = pd.DataFrame(content["psi_ones"], columns=["Feature name", "PSI value"]).to_html(
            index=False
        )

        env = Environment(loader=FileSystemLoader(searchpath=self.template_path))
        self._sections["whitebox"] = env.get_template(self._whitebox_section_path).render(content)


def plot_data_hist(data, title="title", bins=100, path=None):
    sns.set(style="whitegrid", font_scale=1.5)
    fig, axs = plt.subplots(figsize=(16, 10))
    sns.distplot(data, bins=bins, color="m", ax=axs)
    axs.set_title(title)
    fig.savefig(path, bbox_inches="tight")
    plt.close()


class ReportDecoNLP(ReportDeco):
    """
    Special report wrapper for :class:`~lightautoml.automl.presets.text_presets.TabularNLPAutoML`.
    Usage case is the same as main
    :class:`~lightautoml.report.report_deco.ReportDeco` class.
    It generates same report as :class:`~lightautoml.report.report_deco.ReportDeco` ,
    but with additional NLP report part.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._nlp_section_path = "nlp_section.html"
        self._nlp_subsection_path = "nlp_subsection.html"
        self._nlp_subsections = []
        self.sections_order.append("nlp")

    def __call__(self, model):
        self._model = model

        # add informataion to report
        self._model_name = model.__class__.__name__
        self._model_parameters = json2html.convert(extract_params(model))
        self._model_summary = None

        self._sections = {}
        self._sections["intro"] = "<p>This report was generated automatically.</p>"
        self._model_results = []
        self._n_test_sample = 0

        self._generate_model_section()
        self.generate_report()
        return self

    def fit_predict(self, *args, **kwargs):
        """Wrapped :meth:`TabularNLPAutoML.fit_predict` method.

        Valid args, kwargs are the same as wrapped automl.

        Args:
            *args: Arguments.
            **kwargs: Additional parameters.

        Returns:
            OOF predictions.

        """
        preds = super().fit_predict(*args, **kwargs)

        train_data = kwargs["train_data"] if "train_data" in kwargs else args[0]
        roles = kwargs["roles"] if "roles" in kwargs else args[1]

        self._text_fields = roles["text"]
        for text_field in self._text_fields:
            content = {}
            content["title"] = "Text field: " + text_field
            content["char_len_hist"] = text_field + "_char_len_hist.png"
            plot_data_hist(
                data=train_data[text_field].apply(len).values,
                path=os.path.join(self.output_path, content["char_len_hist"]),
                title="Length in char",
            )
            content["tokens_len_hist"] = text_field + "_tokens_len_hist.png"
            plot_data_hist(
                data=train_data[text_field].str.split(" ").apply(len).values,
                path=os.path.join(self.output_path, content["tokens_len_hist"]),
                title="Length in tokens",
            )
            self._generate_nlp_subsection(content)
        # Concatenated text fields
        if len(self._text_fields) >= 2:
            all_fields = train_data[self._text_fields].agg(" ".join, axis=1)
            content = {}
            content["title"] = "Concatenated text fields"
            content["char_len_hist"] = "concat_char_len_hist.png"
            plot_data_hist(
                data=all_fields.apply(len).values,
                path=os.path.join(self.output_path, content["char_len_hist"]),
                title="Length in char",
            )
            content["tokens_len_hist"] = "concat_tokens_len_hist.png"
            plot_data_hist(
                data=all_fields.str.split(" ").apply(len).values,
                path=os.path.join(self.output_path, content["tokens_len_hist"]),
                title="Length in tokens",
            )
            self._generate_nlp_subsection(content)

        self._generate_nlp_section()
        self.generate_report()
        return preds

    def _generate_nlp_subsection(self, content):
        # content has the following fields:
        # title:            subsection title
        # char_len_hist:    path to histogram of text length (number of chars)
        # tokens_len_hist:  path to histogram of text length (number of tokens)
        env = Environment(loader=FileSystemLoader(searchpath=self.template_path))
        nlp_subsection = env.get_template(self._nlp_subsection_path).render(content)
        self._nlp_subsections.append(nlp_subsection)

    def _generate_nlp_section(self):
        if self._model_results:
            env = Environment(loader=FileSystemLoader(searchpath=self.template_path))
            nlp_section = env.get_template(self._nlp_section_path).render(nlp_subsections=self._nlp_subsections)
            self._sections["nlp"] = nlp_section


def get_uplift_data(test_target, uplift_pred, test_treatment, mode):
    perfect = uplift_metrics.perfect_uplift_curve(test_target, test_treatment)
    xs, ys = uplift_metrics.calculate_graphic_uplift_curve(test_target, uplift_pred, test_treatment, mode)
    xs_perfect, ys_perfect = uplift_metrics.calculate_graphic_uplift_curve(test_target, perfect, test_treatment, mode)
    uplift_auc = uplift_metrics.calculate_uplift_auc(test_target, uplift_pred, test_treatment, mode, normed=True)
    return xs, ys, xs_perfect, ys_perfect, uplift_auc


def plot_uplift_curve(test_target, uplift_pred, test_treatment, path):
    sns.set(style="whitegrid", font_scale=1.5)
    # plt.figure(figsize=(10, 10));
    fig, axs = plt.subplots(3, 1, figsize=(10, 30))
    # qini
    xs, ys, xs_perfect, ys_perfect, uplift_auc = get_uplift_data(test_target, uplift_pred, test_treatment, mode="qini")
    axs[0].plot(xs, ys, color="blue", lw=2, label="qini mode")
    axs[0].plot(xs_perfect, ys_perfect, color="black", lw=1, label="perfect uplift")
    axs[0].plot(
        (0, xs[-1]),
        (0, ys[-1]),
        color="black",
        lw=1,
        linestyle="--",
        label="random model",
    )
    axs[0].set_title("Uplift qini, AUC={:.3f}".format(uplift_auc))
    axs[0].legend(loc="lower right")
    # cum_gain
    xs, ys, xs_perfect, ys_perfect, uplift_auc = get_uplift_data(
        test_target, uplift_pred, test_treatment, mode="cum_gain"
    )
    axs[1].plot(xs, ys, color="red", lw=2, label="cum_gain model")
    axs[1].plot(xs_perfect, ys_perfect, color="black", lw=1, label="perfect uplift")
    axs[1].plot(
        (0, xs[-1]),
        (0, ys[-1]),
        color="black",
        lw=1,
        linestyle="--",
        label="random model",
    )
    axs[1].set_title("Uplift cum_gain, AUC={:.3f}".format(uplift_auc))
    axs[1].legend(loc="lower right")
    # adj_qini
    xs, ys, xs_perfect, ys_perfect, uplift_auc = get_uplift_data(
        test_target, uplift_pred, test_treatment, mode="adj_qini"
    )
    axs[2].plot(xs, ys, color="green", lw=2, label="adj_qini mode")
    axs[2].plot(xs_perfect, ys_perfect, color="black", lw=1, label="perfect uplift")
    axs[2].plot(
        (0, xs[-1]),
        (0, ys[-1]),
        color="black",
        lw=1,
        linestyle="--",
        label="random model",
    )
    axs[2].set_title("Uplift adj_qini, AUC={:.3f}".format(uplift_auc))
    axs[2].legend(loc="lower right")

    plt.savefig(path, bbox_inches="tight")
    plt.close()


class ReportDecoUplift(ReportDeco):
    _available_metalearners = (TLearner, XLearner)

    @property
    def reader(self):
        if self._is_xlearner:
            return self._model.learners["outcome"]["treatment"].reader  # effect
        else:
            return self._model.treatment_learner.reader

    @property
    def task(self):
        if self._is_xlearner:
            return "reg"
        else:
            return self.reader.task._name

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._uplift_section_path = "uplift_section.html"
        self._uplift_subsection_path = "uplift_subsection.html"
        self.sections_order.append("uplift")
        self._uplift_results = []

    def __call__(self, model):
        self._model = model
        self._is_xlearner = isinstance(model, XLearner)

        # add informataion to report
        self._model_name = model.__class__.__name__
        self._model_parameters = json2html.convert(extract_params(model))
        self._model_summary = None

        self._sections = {}
        self._sections["intro"] = "<p>This report was generated automatically.</p>"
        self._model_results = []
        self._n_test_sample = 0

        self._generate_model_section()
        self.generate_report()
        return self

    def fit(self, *args, **kwargs):
        """Wrapped automl.fit_predict method.
        Valid args, kwargs are the same as wrapped automl.
        Args:
            *args: arguments.
            **kwargs: additional parameters.
        Returns:
            oof predictions.
        """
        train_data = kwargs["train_data"] if "train_data" in kwargs else args[0]
        input_roles = kwargs["roles"] if "roles" in kwargs else args[1]
        self._target = input_roles["target"]
        self._treatment_col = input_roles["treatment"]
        if self._is_xlearner:
            self._fit_xlearner(train_data, input_roles)
        else:
            self._fit_tlearner(train_data, input_roles)
        self._model._is_fitted = True
        self._generate_model_section()
        self._train_data_overview = self._data_general_info(train_data, "train")
        self._describe_roles(train_data)
        self._describe_dropped_features(train_data)
        self._generate_train_set_section()
        self.generate_report()

    def predict(self, test_data):

        """Wrapped tlearner.predict method.
        Valid args, kwargs are the same as wrapped automl.
        Args:
            test_data: Dataset to perform inference.
        Returns:
            predictions.
        """
        self._n_test_sample += 1

        # get predictions
        test_target = test_data[self._target].values
        test_treatment = test_data[self._treatment_col].values
        # test_data = test_data.drop([self._target, self._treatment_col], axis=1)

        uplift, treatment_preds, control_preds = self._model.predict(test_data)

        if self._n_test_sample >= 2:
            treatment_title = "Treatment test {}".format(self._n_test_sample)
            control_title = "Control test {}".format(self._n_test_sample)
        else:
            treatment_title = "Treatment test"
            control_title = "Control test"

        # treatment data
        data = pd.DataFrame({"y_true": test_target[test_treatment == 1]})
        data["y_pred"] = treatment_preds[test_treatment == 1]
        data.sort_values("y_pred", ascending=False, inplace=True)
        data["bin"] = (np.arange(data.shape[0]) / data.shape[0] * self.n_bins).astype(int)
        data = data[~data["y_pred"].isnull()]
        self._generate_test_subsection(data, "treatment", treatment_title)
        self._generate_inference_section(data)

        # control data
        data = pd.DataFrame({"y_true": test_target[test_treatment == 0]})
        data["y_pred"] = control_preds[test_treatment == 0]
        data.sort_values("y_pred", ascending=False, inplace=True)
        data["bin"] = (np.arange(data.shape[0]) / data.shape[0] * self.n_bins).astype(int)
        data = data[~data["y_pred"].isnull()]
        self._generate_test_subsection(data, "control", control_title)
        self._generate_inference_section(data)

        # update model section
        self._generate_model_section()

        # uplift section
        self._uplift_content = {}
        if self._n_test_sample >= 2:
            self._uplift_content["title"] = "Test sample {}".format(self._n_test_sample)
            self._uplift_content["uplift_curve"] = "uplift_curve_{}.png".format(self._n_test_sample)
            self._uplift_content["uplift_distribution"] = "uplift_distribution_{}.png".format(self._n_test_sample)
        else:
            self._uplift_content["title"] = "Test sample"
            self._uplift_content["uplift_curve"] = "uplift_curve.png"
            self._uplift_content["uplift_distribution"] = "uplift_distribution.png"
        plot_uplift_curve(
            test_target,
            uplift,
            test_treatment,
            path=os.path.join(self.output_path, self._uplift_content["uplift_curve"]),
        )
        self._uplift_distribution(
            test_target,
            uplift,
            test_treatment,
            path=os.path.join(self.output_path, self._uplift_content["uplift_distribution"]),
        )

        self._uplift_content["test_data_overview"] = self._data_general_info(test_data, "test")

        self._generate_uplift_subsection()
        self._generate_uplift_section()

        self.generate_report()
        return uplift, treatment_preds, control_preds

    def _uplift_distribution(self, test_target, uplift, test_treatment, path):
        data = pd.DataFrame({"y_true": test_target, "y_pred": uplift, "treatment": test_treatment})
        data.sort_values("y_pred", ascending=True, inplace=True)
        data["bin"] = (np.arange(data.shape[0]) / data.shape[0] * self.n_bins).astype(int)
        # 'Uplift fact'
        mean_target_treatment = (
            data[data["treatment"].values == 1].groupby("bin").agg({"y_true": [np.mean]}).values[:, 0]
        )
        mean_target_control = data[data["treatment"].values == 0].groupby("bin").agg({"y_true": [np.mean]}).values[:, 0]
        uplift_fact = mean_target_treatment - mean_target_control
        bins_table = data.groupby("bin").agg({"y_true": [len], "y_pred": [np.min, np.mean, np.max]}).reset_index()
        bins_table.columns = [
            "Bin number",
            "Amount of objects",
            "Min uplift",
            "Mean uplift",
            "Max uplift",
        ]
        bins_table["Uplift fact"] = uplift_fact
        self._uplift_content["uplift_bins_table"] = bins_table.to_html(index=False)

        # uplift kde distribution
        sns.set(style="whitegrid", font_scale=1.5)
        fig, axs = plt.subplots(figsize=(16, 10))

        sns.kdeplot(data["y_pred"], shade=True, color="g", label="y_pred", ax=axs)
        axs.set_xlabel("Uplift value")
        axs.set_ylabel("Density")
        axs.set_title("Uplift distribution")

        fig.savefig(path, bbox_inches="tight")
        plt.close()

    def _fit_tlearner(self, train_data, roles):
        treatment_role, _ = _get_treatment_role(roles)
        new_roles = deepcopy(roles)
        new_roles.pop(treatment_role)
        self._model._timer._timeout = 1e10
        self._model._timer.start()
        # treatment
        treatment_train_data = train_data[train_data[self._treatment_col] == 1]
        treatment_target = treatment_train_data[self._target].values
        treatment_train_data.drop(self._treatment_col, axis=1, inplace=True)
        treatment_preds = self._model.treatment_learner.fit_predict(treatment_train_data, new_roles)
        # control
        control_train_data = train_data[train_data[self._treatment_col] == 0]
        control_target = control_train_data[self._target].values
        control_train_data.drop(self._treatment_col, axis=1, inplace=True)
        control_preds = self._model.control_learner.fit_predict(control_train_data, new_roles)

        self._generate_fit_section(treatment_preds, control_preds, treatment_target, control_target)

    def _fit_xlearner(self, train_data, roles):
        treatment_role, _ = _get_treatment_role(roles)
        new_roles = deepcopy(roles)
        new_roles.pop(treatment_role)

        self._model._timer._timeout = 1e10
        self._model._timer.start()
        self._model._fit_propensity_learner(train_data, roles)
        self._model._fit_outcome_learners(train_data, roles)

        # treatment
        treatment_train_data = train_data[train_data[self._treatment_col] == 1]
        treatment_train_data.drop(self._treatment_col, axis=1, inplace=True)
        outcome_pred = self._model.learners["outcome"]["control"].predict(treatment_train_data).data.ravel()
        treatment_train_data[self._target] = treatment_train_data[self._target] - outcome_pred
        treatment_target = treatment_train_data[self._target].values
        treatment_preds = self._model.learners["effect"]["treatment"].fit_predict(treatment_train_data, new_roles)

        # control
        control_train_data = train_data[train_data[self._treatment_col] == 0]
        control_train_data.drop(self._treatment_col, axis=1, inplace=True)
        outcome_pred = self._model.learners["outcome"]["treatment"].predict(control_train_data).data.ravel()
        control_train_data[self._target] = control_train_data[self._target] - outcome_pred
        control_train_data[self._target] *= -1
        control_target = control_train_data[self._target].values
        control_preds = self._model.learners["effect"]["control"].fit_predict(control_train_data, new_roles)

        self._generate_fit_section(treatment_preds, control_preds, treatment_target, control_target)

    def _generate_fit_section(self, treatment_preds, control_preds, treatment_target, control_target):
        self._generate_model_summary_table()
        # treatment model
        treatment_data = self._collect_data(treatment_preds, treatment_target)
        self._generate_training_subsection(treatment_data, "treatment", "Treatment train")
        self._generate_inference_section(treatment_data)

        control_data = self._collect_data(control_preds, control_target)
        self._generate_training_subsection(control_data, "control", "Control train")
        self._generate_inference_section(control_data)

    def _collect_data(self, preds, target):
        data = pd.DataFrame({"y_true": target})
        if self.task in "multiclass":
            if self.mapping is not None:
                data["y_true"] = np.array([self.mapping[y] for y in data["y_true"].values])
            data["y_pred"] = preds._data.argmax(axis=1)
        else:
            data["y_pred"] = preds._data[:, 0]
            data.sort_values("y_pred", ascending=False, inplace=True)
            data["bin"] = (np.arange(data.shape[0]) / data.shape[0] * self.n_bins).astype(int)
        # remove NaN in predictions:
        data = data[~data["y_pred"].isnull()]
        return data

    def _generate_model_summary_table(self):
        if self.task == "binary":
            evaluation_parameters = ["AUC-score", "Precision", "Recall", "F1-score"]
            self._model_summary = pd.DataFrame({"Evaluation parameter": evaluation_parameters})
        elif self.task == "reg":
            evaluation_parameters = [
                "Mean absolute error",
                "Median absolute error",
                "Mean squared error",
                "R^2 (coefficient of determination)",
                "Explained variance",
            ]
            self._model_summary = pd.DataFrame({"Evaluation parameter": evaluation_parameters})

    def _generate_training_subsection(self, data, prefix, title):
        self._inference_content = {}
        self._inference_content["title"] = title
        if self.task == "binary":
            # filling for html
            self._inference_content["roc_curve"] = prefix + "_roc_curve.png"
            self._inference_content["pr_curve"] = prefix + "_pr_curve.png"
            self._inference_content["pie_f1_metric"] = prefix + "_pie_f1_metric.png"
            self._inference_content["preds_distribution_by_bins"] = prefix + "_preds_distribution_by_bins.png"
            self._inference_content["distribution_of_logits"] = prefix + "_distribution_of_logits.png"
            # graphics and metrics
            _, self._F1_thresh = f1_score_w_co(data)
            self._model_summary[title] = self._binary_classification_details(data)
        elif self.task == "reg":
            # filling for html
            self._inference_content["target_distribution"] = prefix + "_target_distribution.png"
            self._inference_content["error_hist"] = prefix + "_error_hist.png"
            self._inference_content["scatter_plot"] = prefix + "_scatter_plot.png"
            # graphics and metrics
            self._model_summary[title] = self._regression_details(data)

    def _generate_test_subsection(self, data, prefix, title):
        self._inference_content = {}
        self._inference_content["title"] = title
        if self.task == "binary":
            # filling for html
            self._inference_content["roc_curve"] = prefix + "_roc_curve_{}.png".format(self._n_test_sample)
            self._inference_content["pr_curve"] = prefix + "_pr_curve_{}.png".format(self._n_test_sample)
            self._inference_content["pie_f1_metric"] = prefix + "_pie_f1_metric_{}.png".format(self._n_test_sample)
            self._inference_content["bins_preds"] = prefix + "_bins_preds_{}.png".format(self._n_test_sample)
            self._inference_content[
                "preds_distribution_by_bins"
            ] = prefix + "_preds_distribution_by_bins_{}.png".format(self._n_test_sample)
            self._inference_content["distribution_of_logits"] = prefix + "_distribution_of_logits_{}.png".format(
                self._n_test_sample
            )
            # graphics and metrics
            self._model_summary[title] = self._binary_classification_details(data)
        elif self.task == "reg":
            # filling for html
            self._inference_content["target_distribution"] = prefix + "_target_distribution_{}.png".format(
                self._n_test_sample
            )
            self._inference_content["error_hist"] = prefix + "_error_hist_{}.png".format(self._n_test_sample)
            self._inference_content["scatter_plot"] = prefix + "_scatter_plot_{}.png".format(self._n_test_sample)
            # graphics
            self._model_summary[title] = self._regression_details(data)

    def _data_general_info(self, data, stage="train"):
        general_info = pd.DataFrame(columns=["Parameter", "Value"])
        general_info.loc[0] = ("Number of records", data.shape[0])
        general_info.loc[1] = ("Share of treatment", np.mean(data[self._treatment_col]))
        general_info.loc[2] = ("Mean target", np.mean(data[self._target]))
        general_info.loc[3] = (
            "Mean target on treatment",
            np.mean(data[self._target][data[self._treatment_col] == 1]),
        )
        general_info.loc[4] = (
            "Mean target on control",
            np.mean(data[self._target][data[self._treatment_col] == 0]),
        )
        if stage == "train":
            general_info.loc[5] = ("Total number of features", data.shape[1])
            general_info.loc[6] = ("Used features", len(self.reader._used_features))
            dropped_list = [col for col in self.reader._dropped_features if col != self._target]
            general_info.loc[7] = ("Dropped features", len(dropped_list))
        return general_info.to_html(index=False, justify="left")

    def _describe_roles(self, train_data):

        # detect feature roles
        # roles = self._model.reader._roles
        roles = self.reader._roles
        numerical_features = [feat_name for feat_name in roles if roles[feat_name].name == "Numeric"]
        categorical_features = [feat_name for feat_name in roles if roles[feat_name].name == "Category"]
        datetime_features = [feat_name for feat_name in roles if roles[feat_name].name == "Datetime"]

        # numerical roles
        numerical_features_df = []
        for feature_name in numerical_features:
            item = {"Feature name": feature_name}
            item["NaN ratio"] = "{:.4f}".format(train_data[feature_name].isna().sum() / train_data.shape[0])
            values = train_data[feature_name].dropna().values
            item["min"] = np.min(values)
            item["quantile_25"] = np.quantile(values, 0.25)
            item["average"] = np.mean(values)
            item["median"] = np.median(values)
            item["quantile_75"] = np.quantile(values, 0.75)
            item["max"] = np.max(values)
            numerical_features_df.append(item)
        if numerical_features_df == []:
            self._numerical_features_table = None
        else:
            self._numerical_features_table = pd.DataFrame(numerical_features_df).to_html(
                index=False, float_format="{:.2f}".format, justify="left"
            )
        # categorical roles
        categorical_features_df = []
        for feature_name in categorical_features:
            item = {"Feature name": feature_name}
            item["NaN ratio"] = "{:.4f}".format(train_data[feature_name].isna().sum() / train_data.shape[0])
            value_counts = train_data[feature_name].value_counts(normalize=True)
            values = value_counts.index.values
            counts = value_counts.values
            item["Number of unique values"] = len(counts)
            item["Most frequent value"] = values[0]
            item["Occurance of most frequent"] = "{:.1f}%".format(100 * counts[0])
            item["Least frequent value"] = values[-1]
            item["Occurance of least frequent"] = "{:.1f}%".format(100 * counts[-1])
            categorical_features_df.append(item)
        if categorical_features_df == []:
            self._categorical_features_table = None
        else:
            self._categorical_features_table = pd.DataFrame(categorical_features_df).to_html(
                index=False, justify="left"
            )
        # datetime roles
        datetime_features_df = []
        for feature_name in datetime_features:
            item = {"Feature name": feature_name}
            item["NaN ratio"] = "{:.4f}".format(train_data[feature_name].isna().sum() / train_data.shape[0])
            values = train_data[feature_name].dropna().values
            item["min"] = np.min(values)
            item["max"] = np.max(values)
            item["base_date"] = self.reader._roles[feature_name].base_date
            datetime_features_df.append(item)
        if datetime_features_df == []:
            self._datetime_features_table = None
        else:
            self._datetime_features_table = pd.DataFrame(datetime_features_df).to_html(index=False, justify="left")

    def _describe_dropped_features(self, train_data):
        self._max_nan_rate = self.reader.max_nan_rate
        self._max_constant_rate = self.reader.max_constant_rate
        self._features_dropped_list = self.reader._dropped_features
        # dropped features table
        dropped_list = [col for col in self._features_dropped_list if col != self._target]
        if dropped_list == []:
            self._dropped_features_table = None
        else:
            dropped_nan_ratio = train_data[dropped_list].isna().sum() / train_data.shape[0]
            dropped_most_occured = pd.Series(np.nan, index=dropped_list)
            for col in dropped_list:
                col_most_occured = train_data[col].value_counts(normalize=True).values
                if len(col_most_occured) > 0:
                    dropped_most_occured[col] = col_most_occured[0]
            dropped_features_table = pd.DataFrame(
                {"nan_rate": dropped_nan_ratio, "constant_rate": dropped_most_occured}
            )
            self._dropped_features_table = (
                dropped_features_table.reset_index()
                .rename(columns={"index": "Название переменной"})
                .to_html(index=False, justify="left")
            )

    def _generate_uplift_subsection(self):
        env = Environment(loader=FileSystemLoader(searchpath=self.template_path))
        uplift_subsection = env.get_template(self._uplift_subsection_path).render(self._uplift_content)
        self._uplift_results.append(uplift_subsection)

    def _generate_uplift_section(self):
        if self._model_results:
            env = Environment(loader=FileSystemLoader(searchpath=self.template_path))
            results_section = env.get_template(self._uplift_section_path).render(uplift_results=self._uplift_results)
            self._sections["uplift"] = results_section
