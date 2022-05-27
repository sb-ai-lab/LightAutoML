from typing import Union, List

from lightautoml.dataset.roles import NumericRole, Dtype
import numpy as np


class NumericVectorOrArrayRole(NumericRole):
    """
    Role that describe numeric vector or numeric array.
    """

    _name = "NumericOrArrayVector"

    def __init__(
            self,
            size: int,
            element_col_name_template: Union[str, List[str]],
            dtype: Dtype = np.float32,
            force_input: bool = False,
            prob: bool = False,
            discretization: bool = False,
            is_vector: bool = True
    ):
        """
        Args:
            size: number of elements in every vector in this column
            element_col_name_template: string template to produce name for each element in the vector
            when array-to-columns transformation is neccessary
            dtype: type of the vector's elements
            force_input: Select a feature for training,
              regardless of the selector results.
            prob: If input number is probability.
        """
        super().__init__(dtype, force_input, prob, discretization)
        self._size = size
        self._element_col_name_template = element_col_name_template
        self._is_vector = is_vector

    @property
    def size(self):
        return self._size

    @property
    def is_vector(self):
        return self._is_vector

    def feature_name_at(self, position: int) -> str:
        """
        produces a name for feature on ``position`` in the vector

        Args:
            position: position in the vector in range [0 .. size]

        Returns:
            feature name

        """
        assert 0 <= position < self.size

        if isinstance(self._element_col_name_template, str):
            return self._element_col_name_template.format(position)

        return self._element_col_name_template[position]
