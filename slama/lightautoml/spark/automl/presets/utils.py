import calendar
import datetime
import logging
from typing import Iterator, List, Tuple
import numpy as np

import pandas as pd

from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql.types import StructField

from lightautoml.spark.utils import SparkDataFrame
from lightautoml.spark.tasks.base import SparkMetric

logger = logging.getLogger(__name__)

MAX_DAY = {
    1: 31,
    2: 28,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31,
}


def replace_year_in_date(date: datetime.datetime, year: int):
    if date.month == 2 and date.day == 29 and not calendar.isleap(year):
        date -= pd.Timedelta(1, "d")
    return date.replace(year=year)


def replace_month_in_date(date: datetime.datetime, month: int):
    if date.day > MAX_DAY[month]:
        date -= pd.Timedelta(date.day - MAX_DAY[month], "d")
        if month == 2 and date.day == 28 and calendar.isleap(date.year):
            date += pd.Timedelta(1, "d")
    return date.replace(month=month)


def replace_dayofweek_in_date(date: datetime.datetime, dayofweek: int):
    date += pd.Timedelta(dayofweek - date.weekday(), "d")
    return date


def calc_feats_permutation_imps(model,
                                used_feats: List[str],
                                data: SparkDataFrame,
                                metric: SparkMetric,
                                silent: bool = False):
    n_used_feats = len(used_feats)
    if not silent:
        logger.info3("LightAutoML used {} feats".format(n_used_feats))

    preds = model.predict(data, add_reader_attrs=True)
    norm_score = metric(preds)

    feat_imp = []
    for it, feat in enumerate(used_feats):
        feat_imp.append(
            calc_one_feat_imp(
                (it + 1, n_used_feats),
                feat,
                model,
                data,
                norm_score,
                metric,
                silent
            )
        )
    feat_imp = pd.DataFrame(feat_imp, columns=["Feature", "Importance"])
    feat_imp = feat_imp.sort_values("Importance", ascending=False).reset_index(drop=True)
    return feat_imp


def calc_one_feat_imp(iters: Tuple[int, int],
                      feat: str,
                      model,
                      data: SparkDataFrame,
                      norm_score: float,
                      metric: SparkMetric,
                      silent: bool,
                      seed: int = 42):

    field: StructField = data.schema[feat]

    @pandas_udf(returnType=field.dataType)
    def permutate(arrs: Iterator[pd.Series]) -> Iterator[pd.Series]:
        permutator = np.random.RandomState(seed=seed)
        # one may get list of arrs and concatenate them to perform permutation
        # in the whole partition
        for x in arrs:
            px = permutator.permutation(x)
            yield pd.Series(px)

    permutated_df = data.withColumn(feat, permutate(feat))

    preds = model.predict(permutated_df, add_reader_attrs=True)
    new_score = metric(preds)

    if not silent:
        logger.info3("{}/{} Calculated score for {}: {:.7f}".format(iters[0], iters[1], feat, norm_score - new_score))

    return feat, norm_score - new_score
