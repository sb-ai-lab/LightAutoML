"""Exercise the optional feature generator with its compatible dependency set."""

import warnings

import numpy as np
import pandas as pd
import pytest


# Woodwork 0.31 still imports this deprecated API. Keep the exception specific
# so other warnings in feature generation remain errors in the compatibility CI.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API", category=UserWarning)
    ft = pytest.importorskip("featuretools")

from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.roles import NumericRole
from lightautoml.transformers.generator import FeatureGeneratorTransformer


def test_feature_generator_creates_entities_and_computes_features():
    data = pd.DataFrame({"value": [-1.0, 2.0, -3.0]})
    dataset = PandasDataset(data, roles={"value": NumericRole()})
    generator = FeatureGeneratorTransformer(
        seq_params={}, agg_primitives=[], trans_primitives=["absolute"], max_depth=1
    )
    generator._create_entities(dataset)
    generator._ignore_columns = [generator.main_id]
    features = generator._get_new_feature_names()
    matrix = ft.calculate_feature_matrix(features=features, entityset=generator.es, n_jobs=1)
    np.testing.assert_array_equal(matrix["ABSOLUTE(value)"], [1, 2, 3])
