"""Exercise the optional feature generator with its compatible dependency set."""

import pickle
import subprocess
import sys
import textwrap
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
from lightautoml.dataset.roles import CategoryRole
from lightautoml.dataset.roles import DatetimeRole
from lightautoml.dataset.roles import NumericRole
from lightautoml.dataset.seq_np_pd_dataset import SeqNumpyPandasDataset
from lightautoml.tasks import Task
from lightautoml.transformers.generator import FeatureGeneratorTransformer
from lightautoml.utils import _woodwork


# Featuretools 1.31 supplies np.sum to groupby. Pandas 2 warns about its own
# future dispatch change; pandas 3 already implements it. Fixing this upstream
# call would require changing Featuretools aggregation semantics, outside this
# accessor adapter. Keep the existing pandas 2 behavior and exempt only that
# warning in tests, without changing application warning filters.
if pd.__version__.startswith("2."):
    pytestmark = pytest.mark.filterwarnings(
        "ignore:The provided callable .* is currently using SeriesGroupBy.sum:FutureWarning:featuretools.computational_backends.feature_set_calculator"
    )


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


@pytest.mark.parametrize("dtype", ["float64", "Int64", "Float64"])
def test_nullable_features_and_accessor_state(dtype):
    _woodwork.initialize_woodwork()
    data = pd.DataFrame({"id": [0, 1, 2], "value": pd.Series([-1, None, 3], dtype=dtype)})
    es = ft.EntitySet(id="nullable").add_dataframe(dataframe_name="values", dataframe=data, index="id")
    accessor = data.ww
    assert data.ww is accessor
    accessor.metadata["source"] = "test"
    assert data.ww.metadata["source"] == "test"
    series = data.ww["value"]
    assert series.ww is series.ww
    assert series.ww.logical_type == data.ww.logical_types["value"]
    feature = ft.TransformFeature(ft.Feature(data.ww["value"]), ft.primitives.Absolute())
    for entityset in (es, pickle.loads(pickle.dumps(es))):
        matrix = ft.calculate_feature_matrix([feature], entityset=entityset)
        np.testing.assert_allclose(matrix["ABSOLUTE(value)"].to_numpy(dtype=float), [1, np.nan, 3], equal_nan=True)


@pytest.mark.parametrize(
    "pandas_version,woodwork_version", [("2.3.3", "0.31.0"), ("3.0.6", "0.32.0"), ("4.0.0", "0.31.0")]
)
def test_adapter_leaves_unaffected_versions_unchanged(monkeypatch, pandas_version, woodwork_version):
    descriptors = [vars(owner)["ww"] for owner in (pd.DataFrame, pd.Series)]
    monkeypatch.setattr(pd, "__version__", pandas_version)
    monkeypatch.setattr(_woodwork, "version", lambda name: woodwork_version)
    assert not _woodwork.initialize_woodwork()
    assert descriptors == [vars(owner)["ww"] for owner in (pd.DataFrame, pd.Series)]


def test_adapter_is_lazy_idempotent_and_only_changes_woodwork():
    # A fresh interpreter makes this independent of collection and test order.
    code = textwrap.dedent(
        """
        import warnings
        warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API", category=UserWarning)
        import pandas as pd
        import woodwork
        before = {owner: dict(vars(owner)) for owner in (pd.DataFrame, pd.Series)}
        from lightautoml.transformers.generator import FeatureGeneratorTransformer
        for owner in before:
            assert vars(owner)["ww"] is before[owner]["ww"]
        FeatureGeneratorTransformer(seq_params={})
        installed = [vars(owner)["ww"] for owner in before]
        FeatureGeneratorTransformer(seq_params={})
        assert installed == [vars(owner)["ww"] for owner in before]
        for owner, attributes in before.items():
            changed = {name for name, value in attributes.items() if vars(owner).get(name) is not value}
            assert changed == ({"ww"} if pd.__version__.startswith("3.") else set())
            assert set(vars(owner)) == set(attributes)
    """
    )
    subprocess.run([sys.executable, "-W", "error", "-c", code], check=True, timeout=60)


def test_temporal_features_respect_cutoff():
    _woodwork.initialize_woodwork()
    events = pd.DataFrame({"id": np.arange(4), "time": pd.date_range("2024-01-01 01:00", periods=4, freq="h")})
    es = ft.EntitySet(id="temporal").add_dataframe(
        dataframe_name="events", dataframe=events, index="id", time_index="time"
    )
    time_feature = ft.Feature(es["events"].ww["time"])
    features = [
        ft.TransformFeature(time_feature, primitive)
        for primitive in (ft.primitives.Hour(), ft.primitives.TimeSincePrevious())
    ]
    matrix = ft.calculate_feature_matrix(features, entityset=es, cutoff_time=pd.Timestamp("2024-01-01 03:30"))
    np.testing.assert_array_equal(matrix.index, [0, 1, 2])
    np.testing.assert_array_equal(matrix["HOUR(time)"], [1, 2, 3])
    np.testing.assert_allclose(matrix["TIME_SINCE_PREVIOUS(time)"], [np.nan, 3600, 3600], equal_nan=True)


@pytest.fixture
def relational_dataset():
    count = 60
    parents = pd.DataFrame({"customer_id": np.arange(count)})
    events = pd.DataFrame(
        {
            "customer_id": np.repeat(np.arange(count), 2),
            "amount": np.arange(count * 2, dtype=float) + 1,
            "event_time": pd.date_range("2024-01-01", periods=count * 2, freq="h"),
            "is_credit": np.tile([False, True], count),
        }
    )
    dataset = PandasDataset(
        parents,
        roles={"customer_id": NumericRole(dtype=np.int64)},
        task=Task("reg"),
        target=pd.Series(np.arange(count, dtype=float)),
    )
    dataset.seq_data = {
        "events": SeqNumpyPandasDataset(
            events,
            roles={
                "customer_id": NumericRole(dtype=np.int64),
                "amount": NumericRole(),
                "event_time": DatetimeRole(),
                "is_credit": CategoryRole(dtype=bool),
            },
        )
    }
    return dataset


def _make_generator(n_jobs=1):
    generator = FeatureGeneratorTransformer(
        seq_params={"events": {"scheme": {"to": "plain", "to_id": "customer_id", "from_id": "customer_id"}}},
        agg_primitives=["count", "sum", "num_unique"],
        trans_primitives=["hour"],
        interesting_values={"events": {"is_credit": [True]}},
        max_depth=3,
        n_jobs=n_jobs,
    )
    # Exercise the actual LightGBM selector, with a small deterministic model.
    for algo in (generator.feats_imp.ml_algo, generator.feats_imp._empty_algo):
        algo.default_params.update(num_trees=3, num_threads=1, min_data_in_leaf=2)
    return generator


@pytest.fixture
def fitted_generator(relational_dataset):
    generator = _make_generator()
    generated = generator.fit_transform(relational_dataset)
    assert generator.feats_imp.is_fitted
    assert generated.shape[0] == len(relational_dataset.data)
    assert generated.features == generator.selected_features
    assert np.isfinite(generated.data.to_numpy()).all()
    return generator


def test_full_fit_transform_relational_and_temporal_features(fitted_generator, relational_dataset):
    generator = fitted_generator
    matrix = generator._calculate_feature_matrix(generator._ft_features)
    np.testing.assert_allclose(matrix["plain_customer_id.COUNT(events)"], 2)
    np.testing.assert_allclose(matrix["plain_customer_id.SUM(events.amount)"], np.arange(60) * 4 + 3)
    np.testing.assert_allclose(matrix["plain_customer_id.NUM_UNIQUE(events.HOUR(event_time))"], 2)
    first = generator.transform(relational_dataset)
    second = generator.transform(relational_dataset)
    pd.testing.assert_frame_equal(first.data, second.data)


def test_fitted_generator_pickle_in_fresh_process(fitted_generator, relational_dataset, tmp_path):
    model_path = tmp_path / "generator.pkl"
    data_path = tmp_path / "dataset.pkl"
    expected_path = tmp_path / "expected.pkl"
    model_path.write_bytes(pickle.dumps(fitted_generator))
    data_path.write_bytes(pickle.dumps(relational_dataset))
    fitted_generator.transform(relational_dataset).data.to_pickle(expected_path)
    code = textwrap.dedent(
        """
        import pickle
        import sys
        import warnings
        warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API", category=UserWarning)
        import pandas as pd
        with open(sys.argv[1], "rb") as stream:
            generator = pickle.load(stream)
        if pd.__version__.startswith("2."):
            warnings.filterwarnings("ignore", message="The provided callable .* is currently using SeriesGroupBy.sum", category=FutureWarning, module="featuretools.computational_backends.feature_set_calculator")
        with open(sys.argv[2], "rb") as stream:
            dataset = pickle.load(stream)
        expected = pd.read_pickle(sys.argv[3])
        pd.testing.assert_frame_equal(generator.transform(dataset).data, expected)
    """
    )
    subprocess.run(
        [sys.executable, "-W", "error", "-c", code, str(model_path), str(data_path), str(expected_path)],
        check=True,
        timeout=60,
    )


def test_parallel_feature_generation(relational_dataset):
    pytest.importorskip("distributed")
    generator = _make_generator(n_jobs=2)
    actual = generator.fit_transform(relational_dataset)
    all_parallel = generator._calculate_feature_matrix(generator._ft_features)
    generator.n_jobs = 1
    expected = generator.transform(relational_dataset)
    pd.testing.assert_frame_equal(actual.data, expected.data)
    pd.testing.assert_frame_equal(all_parallel, generator._calculate_feature_matrix(generator._ft_features))


def test_parallel_temporal_features_keep_complete_history():
    pytest.importorskip("distributed")
    generator = FeatureGeneratorTransformer(seq_params={}, n_jobs=2)
    data = pd.DataFrame({"id": [0, 1, 2, 3], "time": pd.date_range("2024-01-01", periods=4, freq="h")})
    generator.es = ft.EntitySet(id="history").add_dataframe(
        dataframe_name="plain", dataframe=data, index="id", time_index="time"
    )
    time_feature = ft.Feature(generator.es["plain"].ww["time"])
    features = [
        ft.TransformFeature(time_feature, primitive)
        for primitive in (ft.primitives.TimeSincePrevious(), ft.primitives.Hour())
    ]
    actual = generator._calculate_feature_matrix(features)
    np.testing.assert_allclose(actual["TIME_SINCE_PREVIOUS(time)"], [np.nan, 3600, 3600, 3600], equal_nan=True)
    generator.n_jobs = 1
    pd.testing.assert_frame_equal(actual, generator._calculate_feature_matrix(features))
