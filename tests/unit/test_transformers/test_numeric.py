
from lightautoml.transformers.numeric import FillnaMean
from lightautoml.transformers.numeric import FillnaMedian


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