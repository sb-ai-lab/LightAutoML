from typing import Optional
from typing import Union

import numpy as np

from ...dataset.np_pd_dataset import NumpyDataset
from ...dataset.np_pd_dataset import PandasDataset
from ...dataset.roles import NumericRole
from ...transformers.base import ChangeRoles
from ...transformers.base import ColumnsSelector
from ...transformers.base import ConvertDataset
from ...transformers.base import LAMLTransformer
from ...transformers.base import SequentialTransformer
from ...transformers.base import UnionTransformer
from ...transformers.categorical import LabelEncoder
from ...transformers.datetime import TimeToNum
from ...transformers.numeric import FillInf
from ...transformers.numeric import FillnaMean
from ...transformers.numeric import QuantileTransformer
from ...transformers.numeric import StandardScaler
from ..utils import get_columns_by_role
from .base import FeaturesPipeline
from .base import TabularDataFeatures


NumpyOrPandas = Union[PandasDataset, NumpyDataset]


class TorchSimpleFeatures(FeaturesPipeline, TabularDataFeatures):
    """Creates simple pipeline for neural network models."""

    def __init__(
        self,
        use_te: bool = False,
        top_intersections: int = 5,
        max_bin_count: int = 10,
        max_intersection_depth: int = 3,
        te_subsample: Optional[Union[int, float]] = None,
        sparse_ohe: Union[str, bool] = "auto",
        auto_unique_co: int = 50,
        output_categories: bool = True,
        multiclass_te_co: int = 3,
        use_qnt: bool = True,
        n_quantiles: Optional[int] = None,
        subsample: int = 1e9,
        output_distribution: str = "normal",
        noise: float = 1e-3,
        qnt_factor: int = 30,
        **kwargs
    ):
        """TorchSimpleFeatures.

        Args:
            use_qnt: Use quantile transformer for numerical columns.
            n_quantiles: Number of quantiles to be computed.
            subsample: Maximum number of samples used to estimate the quantiles for computational efficiency.
            output_distribution: Marginal distribution for the transformed data. The choices are 'uniform' or 'normal'.
            noise: Add noise with certain std to dataset before quantile transformation to make data more smooth.
            qnt_factor: If number of quantiles is none then it equals dataset size / factor
            use_te: Use target encoding for categorical columns.
            top_intersections: Max number of categories to generate intersections.
            max_bin_count: Max number of bins for cat columns.
            max_intersection_depth: Max depth of cat intersection.
            te_subsample: Subsample to calc data statistics
            sparse_ohe: Should we output sparse if ohe encoding was used during cat handling.
            auto_unique_co: Switch to target encoding if high cardinality.
            output_categories: Output encoded categories or embed idxs.
            multiclass_te_co: Cutoff if use target encoding in cat handling on multiclass task if number of classes is high.
        """
        super().__init__(
            multiclass_te=False,
            top_intersections=top_intersections,
            max_intersection_depth=max_intersection_depth,
            subsample=te_subsample,
            auto_unique_co=auto_unique_co,
            output_categories=output_categories,
            ascending_by_cardinality=True,
            max_bin_count=max_bin_count,
            sparse_ohe=sparse_ohe,
            multiclass_te_co=multiclass_te_co,
            **kwargs
        )
        self.use_qnt = use_qnt
        self.n_quantiles = n_quantiles
        self.subsample = subsample
        self.output_distribution = output_distribution
        self.noise = noise
        self.qnt_factor = qnt_factor
        self.use_te = use_te

    def create_pipeline(self, train: NumpyOrPandas) -> LAMLTransformer:
        """Create tree pipeline.

        Args:
            train: Dataset with train features.

        Returns:
            Composite datetime, categorical, numeric transformer.

        """
        transformers_list = []
        cat_cols = get_columns_by_role(train, "Category")
        freq_cols = get_columns_by_role(train, "Category", encoding_type="freq")
        other_cols = set(cat_cols) - set(freq_cols)

        transformers_list.append(self.get_freq_encoding(train, freq_cols))

        # process categories
        if len(other_cols) > 0:
            cat_processing = SequentialTransformer([ColumnsSelector(keys=other_cols), LabelEncoder()])

            if self.use_te:
                target_encoder = self.get_target_encoder(train)

                # get target encoded categories
                te_part = self.get_categorical_raw(train, other_cols)
                if te_part is not None and target_encoder is not None:
                    transformers_list.append(SequentialTransformer([te_part, target_encoder()]))

                # get intersection of top categories
                intersections = self.get_categorical_intersections(train)
                if intersections is not None and target_encoder is not None:
                    transformers_list.append(SequentialTransformer([intersections, target_encoder()]))
            else:
                transformers_list.append(cat_processing)

        # process datetimes
        datetimes = get_columns_by_role(train, "Datetime")
        if len(datetimes) > 0:
            dt_processing = SequentialTransformer([ColumnsSelector(keys=datetimes), TimeToNum()])
            transformers_list.append(dt_processing)

        # process numbers
        numerics = get_columns_by_role(train, "Numeric")
        if len(numerics) > 0:
            num_processing = SequentialTransformer(
                [
                    ColumnsSelector(keys=numerics),
                    FillInf(),
                    FillnaMean(),
                    QuantileTransformer(
                        n_quantiles=self.n_quantiles,
                        subsample=self.subsample,
                        output_distribution=self.output_distribution,
                        noise=self.noise,
                        qnt_factor=self.qnt_factor,
                    )
                    if self.use_qnt
                    else StandardScaler(),
                    ConvertDataset(dataset_type=NumpyDataset),
                    ChangeRoles(NumericRole(np.float32)),
                ]
            )
            transformers_list.append(num_processing)

        union_all = UnionTransformer(transformers_list)

        return union_all
