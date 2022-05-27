from typing import List, Optional

from pyspark.ml.feature import PCA, VectorAssembler
from pyspark.ml.functions import vector_to_array, array_to_vector

from lightautoml.dataset.roles import NumericRole
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.transformers.base import ObsoleteSparkTransformer
from lightautoml.transformers.decomposition import numeric_check

import numpy as np
import pyspark.sql.functions as F


class PCATransformer(ObsoleteSparkTransformer):
    """PCA."""

    _fit_checks = (numeric_check,)
    _transform_checks = ()
    _fname_prefix = "pca"

    @property
    def features(self) -> List[str]:
        """Features list."""
        return self._features

    def __init__(
        self,
        n_components: int = 500
    ):
        """

        Args:
            n_components: Number of PCA components
        """
        # TODO: should we add support for subs and random_state arguments?

        self.n_components = n_components
        self.pca = None
        self.pca_output_col = "pcaFeatures"
        self._features = None

    def _fit(self, dataset: SparkDataset):
        """Fit algorithm on dataset.

        Args:
            dataset: Sparse or Numpy dataset of text features.

        """

        sdf = dataset.data
        self.n_components = np.minimum(self.n_components, len(sdf.columns) - 1)

        sdf = sdf.select(array_to_vector(F.array('*')).alias("features"))
        pca = PCA(k=3, inputCol="features", outputCol=self.pca_output_col)
        self.pca = pca.fit(sdf)

        orig_name = dataset.features[0].split("__")[-1]

        feats = (
            np.char.array([self._fname_prefix + "_"])
            + np.arange(self.n_components).astype(str)
            + np.char.array(["__" + orig_name])
        )

        self._features = list(feats)
        return self

    def _transform(self, dataset: SparkDataset) -> SparkDataset:
        """Transform input dataset to PCA representation.

        Args:
            dataset: Pandas or Numpy dataset of text features.

        Returns:
            Numpy dataset with text embeddings.

        """

        assert self.pca, "This transformer has not been fitted yet"

        sdf = dataset.data.select(*dataset.service_columns, array_to_vector(F.array('*')).alias("features"))
        new_sdf = self.pca\
            .transform(sdf)\
            .select(*dataset.service_columns, *[
                vector_to_array(F.col(self.pca_output_col))[i].alias(feat)
                for i, feat in zip(range(self.n_components), self.features)
            ])

        output = dataset.empty()
        output.set_data(new_sdf, self.features, NumericRole(np.float32))
        return output
