import numpy as np

from lightautoml.transformers.numeric import FillnaMean
from lightautoml.transformers.numeric import FillnaMedian
from lightautoml.transformers.numeric import QuantileTransformer


def test_fillnamean(lamldataset_with_na):
    transformer = FillnaMean()
    output = transformer.fit_transform(lamldataset_with_na)

    assert output.data[:, 0].mean() == 4
    assert output.data[:, 1].mean() == 5
    assert output.data[:, 2].mean() == 0


def test_fillnamedian(lamldataset_with_na):
    transformer = FillnaMedian()
    output = transformer.fit_transform(lamldataset_with_na)

    assert output.data[:, 0].mean() == 4
    assert output.data[:, 1].mean() == 5
    assert output.data[:, 2].mean() == 0


def test_quantiletransformer(lamldataset_30_2):
    transformer = QuantileTransformer(noise=None)
    output = transformer.fit_transform(lamldataset_30_2)

    # raise(Exception(output.data))
    np.testing.assert_allclose(
        output.data,
        np.array(
            [
                [-5.19933758, -5.19933758],
                [-1.47640435, -1.48183072],
                [-1.42177828, -1.44872465],
                [-1.24067307, -1.25262296],
                [-1.02813514, -1.06089913],
                [-0.87314381, -0.95310275],
                [-0.86592145, -0.86396215],
                [-0.62097828, -0.60156557],
                [-0.50478792, -0.5339135],
                [-0.50373715, -0.48136567],
                [-0.39911771, -0.32828215],
                [-0.36893762, -0.30284499],
                [-0.18779519, -0.24328491],
                [-0.12682175, -0.13728361],
                [-0.01319139, 0.02800416],
                [0.01861645, 0.04895583],
                [0.13783602, 0.13759678],
                [0.1464553, 0.23178871],
                [0.2576737, 0.38408782],
                [0.35208669, 0.41496716],
                [0.59203696, 0.44743911],
                [0.6766203, 0.46734882],
                [0.75052545, 0.50159806],
                [0.77324891, 0.80971082],
                [0.95569552, 0.86776588],
                [1.18959458, 1.09445074],
                [1.22743552, 1.26058256],
                [1.43696861, 1.27169708],
                [1.55529186, 1.77127928],
                [5.19933758, 5.19933758],
            ]
        ),
        atol=1e-5,
        rtol=1e-5,
    )
