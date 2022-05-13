"""Generate features for data base structure."""

import logging
import re

from copy import deepcopy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple


try:
    import featuretools as ft
except:
    import warnings

    warnings.warn("'featuretools' - package isn't installed")
import pandas as pd

from ..dataset.base import LAMLDataset
from ..dataset.roles import NumericRole
from ..ml_algo.boost_lgbm import BoostLGBM
from ..pipelines.features.base import TabularDataFeatures
from ..pipelines.features.lgb_pipeline import LGBSimpleFeatures
from ..pipelines.selection.importance_based import ModelBasedImportanceEstimator
from ..pipelines.selection.permutation_importance_based import (
    NpIterativeFeatureSelector,
)
from ..pipelines.utils import get_columns_by_role
from ..reader.utils import set_sklearn_folds
from ..validation.np_iterators import get_numpy_iterator
from ..validation.utils import create_validation_iterator
from .base import LAMLTransformer


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_interesting_values(dataset: LAMLDataset, per_top_categories: float) -> Dict:
    """Get interesting value for featuretools.

    Args:
        dataset: Dataset to search.
        per_top_categories: percent of most frequent categories for feature generation in corresponding slices. If number of unique values is less than 10, then the all values are be used.

    Returns:
        Dictionary of categorical feature names with corresponding unique values.

    """
    unique_values_th = 10

    categorical_features = get_columns_by_role(dataset, "Category")
    cat_interesting_values = dict()
    df = dataset.data

    for feature_name in categorical_features:
        df_stat = df[feature_name].value_counts()
        df_stat_len = len(df_stat)

        if df_stat_len > unique_values_th:
            n_top_categories = round(df_stat_len * per_top_categories * 1e-3)
        else:
            n_top_categories = df_stat_len

        cat_interesting_values[feature_name] = df_stat.index[:n_top_categories].tolist()
    return cat_interesting_values


def renaming_function(column_name) -> str:
    """Renaming for appropriate LGBM format.

    Args:
        column_name: name of column.

    Returns:
        renamed column if initial column contains specific symbols

    """
    return re.sub("[^A-Za-z0-9_()=. ]+", "", column_name)


class FeatureGeneratorTransformer(LAMLTransformer, TabularDataFeatures):
    """Generate features.

    Args:
        seq_params: sequence-related params.
        max_gener_features: maximum generated features.
        max_depth: maximum allowed depth of features.
        agg_primitives: list of aggregation primitives.
        trans_primitives: list of transform primitives.
        interesting_values: categorical values if the form of {table_name: {column: [values]}} for feature generation in corresponding slices.
        generate_interesting_values: whether generate feature in slices of unique categories or not.
        per_top_categories: percent of most frequent categories for feature generation in corresponding slices. If number of unique values is less than 10, then the all values are be used.
        sample_size: size of data to make generated feature selection on it.
        n_jobs: number of processes to run in parallel

    """

    _fit_checks = ()
    _transform_checks = ()
    _fname_prefix = "ft"

    @property
    def features(self) -> List[str]:
        """Features list."""
        return self._features

    def __init__(
        self,
        seq_params: Dict,
        max_gener_features: int = -1,
        max_depth: int = 2,
        agg_primitives: Optional[Tuple[str]] = (
            "entropy",
            "count",
            "mean",
            "std",
            "median",
            "max",
            "sum",
            "num_unique",
            "min",
            "percent_true",
        ),
        trans_primitives: Optional[Tuple[str]] = (
            "hour",
            "month",
            "weekday",
            "is_weekend",
            "day",
            "time_since_previous",
            "week",
            "age",
            "time_since",
        ),
        interesting_values: Dict[str, Dict[str, List[Any]]] = None,
        generate_interesting_values: bool = False,
        per_top_categories: float = 0.25,
        sample_size: int = None,
        n_jobs: int = 1,
    ):

        super().__init__()

        self.seq_params = deepcopy(seq_params)
        self.max_gener_features = max_gener_features
        self.agg_primitives = agg_primitives
        self.trans_primitives = trans_primitives
        self.interesting_values = interesting_values
        self.max_depth = max_depth
        self.generate_interesting_values = generate_interesting_values
        self.per_top_categories = per_top_categories
        self.sample_size = sample_size
        self.n_jobs = n_jobs

        # Make an entityset
        self.es = ft.EntitySet(id="star_schema")
        self.main_id = "plain_unique_index"

        # Initialize feature selector
        self.feats_imp = NpIterativeFeatureSelector(
            feature_pipeline=LGBSimpleFeatures(),
            ml_algo=BoostLGBM(),
            imp_estimator=ModelBasedImportanceEstimator(),
            fit_on_holdout=False,
            max_features_cnt_in_result=None,
        )

        self.seq_table_names = sorted(list(self.seq_params.keys()))

        self._ignore_columns = None
        self._ft_features = None
        self._features = None
        self.selected_features = None
        self.feature_dict = None

    def _create_entities(self, dataset: LAMLDataset):
        """Create entities"""
        # An entity of the main table
        self.es = self.es.add_dataframe(
            dataframe_name="plain",
            dataframe=dataset.data,
            index=self.main_id,  # unique index
            make_index=True,
        )

        # Define unique columns of ids in plain
        plain_unique_ids = list()
        for seq_table_name in self.seq_table_names:
            if self.seq_params[seq_table_name]["scheme"]["to"] == "plain":
                plain_id = self.seq_params[seq_table_name]["scheme"]["to_id"]
                if plain_id not in plain_unique_ids:
                    plain_unique_ids.append(plain_id)
                    self.es.normalize_dataframe(
                        new_dataframe_name=f"plain_{plain_id}",
                        base_dataframe_name="plain",
                        index=plain_id,
                    )

        # Define new tables' names
        # If in the main table contains two keys, there will be two additional tables with the keys and key to connect with major table.
        plain_i_id = dict()
        for unique_id in plain_unique_ids:
            plain_i_id[unique_id] = f"plain_{plain_id}"
        plain_i_id_inv = dict((v, k) for k, v in plain_i_id.items())

        # Update defined params for scheme
        for seq_table_name in self.seq_table_names:
            if self.seq_params[seq_table_name]["scheme"]["to"] == "plain":
                self.seq_params[seq_table_name]["scheme"]["to"] = plain_i_id[
                    self.seq_params[seq_table_name]["scheme"]["to_id"]
                ]

        # Create entities for seq tables
        for seq_table_name in sorted(list(self.seq_params.keys())):
            self.es = self.es.add_dataframe(
                dataframe_name=seq_table_name,
                dataframe=dataset.seq_data[seq_table_name].data,
                index=seq_table_name + "_unique_index",
                make_index=True,
            )

    def _add_relationships(self):
        """Add in the defined relationships"""
        relationships = list()
        for seq_table_name in sorted(list(self.seq_params.keys())):
            scheme = self.seq_params[seq_table_name]["scheme"]
            self.es.add_relationships([(scheme["to"], scheme["to_id"], seq_table_name, scheme["from_id"])])

    def _set_interesting_values(self):
        """Add interesting values if any"""
        for seq_table_name in self.seq_table_names and sorted(list(self.interesting_values.keys())):
            columns = sorted(list(self.interesting_values[seq_table_name].keys()))
            for column in columns:
                values = self.interesting_values[seq_table_name][column]
                self.es[seq_table_name][column].interesting_values = values

    def _get_new_feature_names(self):
        """DFS with specified primitives"""
        return ft.dfs(
            entityset=self.es,
            target_dataframe_name="plain",
            agg_primitives=self.agg_primitives,
            where_primitives=self.agg_primitives,
            trans_primitives=self.trans_primitives,
            max_depth=self.max_depth,
            max_features=self.max_gener_features,
            features_only=True,
            ignore_columns={"plain": self._ignore_columns},
        )

    def fit(self, dataset: LAMLDataset) -> "FeatureToolsTransformer":
        """Extract nan flags.

        Args:
            dataset: Dataset to make transform.

        Returns:
            self.

        """
        # set transformer names and add checks
        for check_func in self._fit_checks:
            check_func(dataset)
        # set transformer features

        # Create entities
        self._create_entities(dataset)

        # Fill dict with new interesting values
        if self.generate_interesting_values:

            new_interesting_values = dict()
            for seq_table_name in self.seq_table_names:
                seq_table = dataset.seq_data[seq_table_name]
                new_interesting_values[seq_table_name] = get_interesting_values(seq_table, self.per_top_categories)

            if self.interesting_values is None:
                self.interesting_values = dict()

            self.interesting_values.update(new_interesting_values)
            logger.info3("Interesting values have been generated")

        # Add interesting values
        if bool(self.interesting_values):
            self._set_interesting_values()
            logger.info3("Interesting values have been added to the entityset")

        # Add relationships
        self._add_relationships()
        logger.info3("Relationships have been added to the entityset")

        # DFS with specified primitives
        self._ignore_columns = dataset.data.columns.tolist()
        self._ft_features = self._get_new_feature_names()

        self._features = [
            self._fname_prefix + "__" + renaming_function(feature_name.get_name()) for feature_name in self._ft_features
        ]
        self.feature_dict = dict(zip(self._features, self._ft_features))

        logger.info3(f"{len(self._features)} are going to be generated")
        return self

    def _select(self, dataset: LAMLDataset) -> List[str]:

        if self.sample_size is None:
            self.sample_size = dataset.shape[0]

        df_sample = dataset.data.sample(self.sample_size, random_state=0)

        self.es.replace_dataframe("plain", df_sample)

        fm_sample = ft.calculate_feature_matrix(features=self._ft_features, entityset=self.es, n_jobs=self.n_jobs)

        fm_sample = fm_sample.set_index(df_sample.index)
        fm_sample = fm_sample.rename(columns=dict(zip(fm_sample.columns, self._features)))

        fm_roles_dict = dict(zip(self._features, [NumericRole()] * len(self._features)))

        target_sample = dataset.target.iloc[df_sample.index]
        folds = set_sklearn_folds(task=dataset.task, target=target_sample.values, cv=20)

        dataset_sample = dataset.empty()
        dataset_sample._initialize(task=dataset.task, folds=pd.Series(folds), target=target_sample)
        dataset_sample.set_data(fm_sample, self._features, fm_roles_dict)

        dataset_sample_iter = create_validation_iterator(dataset_sample)

        self.feats_imp.feature_group_size = max(1, int(len(self._features) * 0.15))  # ~15% of the generated features

        self.feats_imp.fit(dataset_sample_iter)

        logger.info3("Selection completed")
        logger.info3(f"{len(self.feats_imp.selected_features)} features have been selected from generated")

    def transform(self, dataset: LAMLDataset) -> LAMLDataset:
        """Generate new columns from inrintut dataset.

        Args:
            dataset: Dataset to make transform.

        Returns:
            LAMLDataset with new features.

        """
        # checks here
        super().transform(dataset)

        dataset.data["plain_unique_index"] = dataset.data.index

        if self.feats_imp.is_fitted is False:
            self._select(dataset)
            self.selected_features = self.feats_imp.selected_features

        fm_roles_dict = dict(zip(self.selected_features, [NumericRole()] * len(self.selected_features)))

        self.es.replace_dataframe("plain", dataset.data)

        fm = ft.calculate_feature_matrix(
            features=[self.feature_dict.get(feature) for feature in self.selected_features],
            entityset=self.es,
            n_jobs=self.n_jobs,
        )

        fm = fm.rename(columns=dict(zip(fm.columns, self.selected_features)))

        dataset_out = dataset.empty()
        dataset_out.set_data(fm, self.selected_features, fm_roles_dict)

        logger.info3("Feature generation is completed")
        return dataset_out
