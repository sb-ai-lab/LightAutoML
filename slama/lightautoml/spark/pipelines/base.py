from typing import List

from lightautoml.dataset.base import RolesDict


class InputFeaturesAndRoles:
    """
    Class that represents input features and input roles.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._input_roles = None

    @property
    def input_features(self) -> List[str]:
        """Returns list of input features"""
        return list(self.input_roles.keys())

    @property
    def input_roles(self) -> RolesDict:
        """Returns dict of input roles"""
        return self._input_roles

    @input_roles.setter
    def input_roles(self, roles: RolesDict):
        """Sets dict of input roles"""
        assert roles is None or isinstance(roles, dict)
        self._input_roles = roles


class OutputFeaturesAndRoles:
    """
    Class that represents output features and output roles.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._output_roles = None

    @property
    def output_features(self) -> List[str]:
        """Returns list of output features"""
        return list(self.output_roles.keys())

    @property
    def output_roles(self) -> RolesDict:
        """Returns dcit of output roles"""
        return self._output_roles
