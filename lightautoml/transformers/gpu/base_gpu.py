from ..base import SetAttribute

import cudf
import dask_cudf

class SetAttribute_gpu(SetAttribute):
    def __init__ (self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def transform(self, dataset):
        val = dataset[:, self.column].data

        if isinstance(val, cudf.DataFrame):
            val = val.iloc[:, 0]
        elif isinstance(val, dask_cudf.DataFrame):
            val = val[val.columns[0]]
        elif len(val.shape) == 2:
            val = val[:, 0]

        if self.attr not in dataset.__dict__:
            dataset.__dict__[self.attr] = val
            dataset._array_like_attrs.append(self.attr)

        return dataset
