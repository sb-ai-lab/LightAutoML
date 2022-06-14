"""Reader and its derivatives."""

import logging

from copy import deepcopy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import TypeVar
from typing import Union
from typing import cast

import numpy as np
import pandas as pd

from pandas import DataFrame
from pandas import Series

from ..dataset.base import array_attr_roles
from ..dataset.base import valid_array_attributes
from ..dataset.np_pd_dataset import PandasDataset
from ..dataset.roles import CategoryRole
from ..dataset.roles import ColumnRole
from ..dataset.roles import DatetimeRole
from ..dataset.roles import DropRole
from ..dataset.roles import NumericRole
from ..dataset.utils import roles_parser
from ..tasks import Task
from .guess_roles import calc_category_rules
from .guess_roles import calc_encoding_rules
from .guess_roles import get_category_roles_stat
from .guess_roles import get_null_scores
from .guess_roles import get_numeric_roles_stat
from .guess_roles import rule_based_cat_handler_guess
from .guess_roles import rule_based_roles_guess
from .utils import set_sklearn_folds


logger = logging.getLogger(__name__)

# roles, how it's passed to automl
RoleType = TypeVar("RoleType", bound=ColumnRole)
RolesDict = Dict[str, RoleType]

# how user can define roles
UserDefinedRole = Optional[Union[str, RoleType]]

UserDefinedRolesDict = Dict[UserDefinedRole, Sequence[str]]
UserDefinedRolesSequence = Sequence[UserDefinedRole]
UserRolesDefinition = Optional[Union[UserDefinedRole, UserDefinedRolesDict, UserDefinedRolesSequence]]


class Reader:
    """
    Abstract class for analyzing input data and creating inner
    :class:`~lightautoml.dataset.base.LAMLDataset` from raw data.
    Takes data in different formats as input,
    drop obviously useless features,
    estimates avaliable size and returns dataset.

    """

    def __init__(self, task: Task, *args: Any, **kwargs: Any):
        """

        Args:
            task: Task object
            *args: Not used.
            *kwargs: Not used.

        """
        self.task = task
        self._roles = {}
        self._dropped_features = []
        self._used_array_attrs = {}
        self._used_features = []

    @property
    def roles(self) -> RolesDict:
        """Roles dict."""
        return self._roles

    @property
    def dropped_features(self) -> List[str]:
        """List of dropped features."""
        return self._dropped_features

    @property
    def used_features(self) -> List[str]:
        """List of used features."""
        return self._used_features

    @property
    def used_array_attrs(self) -> Dict[str, str]:
        """Dict of used array attributes."""
        return self._used_array_attrs

    def fit_read(
        self,
        train_data: Any,
        features_names: Optional[List[str]] = None,
        roles: UserRolesDefinition = None,
        **kwargs: Any
    ):
        """Abstract function to get dataset with initial feature selection."""
        raise NotImplementedError

    def read(self, data: Any, features_names: Optional[List[str]], **kwargs: Any):
        """Abstract function to add validation columns."""
        raise NotImplementedError

    def upd_used_features(
        self,
        add: Optional[Sequence[str]] = None,
        remove: Optional[Sequence[str]] = None,
    ):
        """Updates the list of used features.

        Args:
            add: List of feature names to add or None.
            remove: List of feature names to remove or None.

        """
        curr_feats = set(self.used_features)
        if add is not None:
            curr_feats = curr_feats.union(add)
        if remove is not None:
            curr_feats = curr_feats - set(remove)
        self._used_features = list(curr_feats)

    @classmethod
    def from_reader(cls, reader: "Reader", **kwargs) -> "Reader":
        """Create reader for new data type from existed.

        Note - for now only Pandas reader exists, made for future plans.

        Args:
            reader: Source reader.
            **kwargs: Ignored as in the class itself.

        Returns:
            New reader.

        """
        new_reader = cls(reader.task, **kwargs)

        for attr in reader.__dict__:
            if attr[0] == "_":
                cls.__dict__[attr] = getattr(reader, attr)

        return new_reader

    def cols_by_type(self, col_type: str) -> List[str]:
        """Get roles names by it's type.

        Args:
            col_type: Column type, for example 'Text'.

        Returns:
            Array with column names.

        """
        names = []
        for col, role in self.roles.items():
            if role.name == col_type:
                names.append(col)

        return names


class PandasToPandasReader(Reader):
    """
    Reader to convert :class:`~pandas.DataFrame` to AutoML's :class:`~lightautoml.dataset.np_pd_dataset.PandasDataset`.
    Stages:

        - Drop obviously useless features.
        - Convert roles dict from user format to automl format.
        - Simple role guess for features without input role.
        - Create cv folds.
        - Create initial PandasDataset.
        - Optional: advanced guessing of role and handling types.

    """

    def __init__(
        self,
        task: Task,
        samples: Optional[int] = 100000,
        max_nan_rate: float = 0.999,
        max_constant_rate: float = 0.999,
        cv: int = 5,
        random_state: int = 42,
        roles_params: Optional[dict] = None,
        n_jobs: int = 4,
        # params for advanced roles guess
        advanced_roles: bool = True,
        numeric_unique_rate: float = 0.999,
        max_to_3rd_rate: float = 1.1,
        binning_enc_rate: float = 2,
        raw_decr_rate: float = 1.1,
        max_score_rate: float = 0.2,
        abs_score_val: float = 0.04,
        drop_score_co: float = 0.01,
        **kwargs: Any
    ):
        """

        Args:
            task: Task object.
            samples: Number of elements used when checking role type.
            max_nan_rate: Maximum nan-rate.
            max_constant_rate: Maximum constant rate.
            cv: CV Folds.
            random_state: Random seed.
            roles_params: dict of params of features roles. \
                Ex. {'numeric': {'dtype': np.float32}, 'datetime': {'date_format': '%Y-%m-%d'}}
                It's optional and commonly comes from config
            n_jobs: Int number of processes.
            advanced_roles: Param of roles guess (experimental, do not change).
            numeric_unqiue_rate: Param of roles guess (experimental, do not change).
            max_to_3rd_rate: Param of roles guess (experimental, do not change).
            binning_enc_rate: Param of roles guess (experimental, do not change).
            raw_decr_rate: Param of roles guess (experimental, do not change).
            max_score_rate: Param of roles guess (experimental, do not change).
            abs_score_val: Param of roles guess (experimental, do not change).
            drop_score_co: Param of roles guess (experimental, do not change).
            **kwargs: For now not used.

        """
        super().__init__(task)
        self.samples = samples
        self.max_nan_rate = max_nan_rate
        self.max_constant_rate = max_constant_rate
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.roles_params = roles_params
        self.target = None
        if roles_params is None:
            self.roles_params = {}

        self.advanced_roles = advanced_roles
        self.advanced_roles_params = {
            "numeric_unique_rate": numeric_unique_rate,
            "max_to_3rd_rate": max_to_3rd_rate,
            "binning_enc_rate": binning_enc_rate,
            "raw_decr_rate": raw_decr_rate,
            "max_score_rate": max_score_rate,
            "abs_score_val": abs_score_val,
            "drop_score_co": drop_score_co,
        }

        self.params = kwargs

    def fit_read(
        self, train_data: DataFrame, features_names: Any = None, roles: UserDefinedRolesDict = None, **kwargs: Any
    ) -> PandasDataset:
        """Get dataset with initial feature selection.

        Args:
            train_data: Input data.
            features_names: Ignored. Just to keep signature.
            roles: Dict of features roles in format
              ``{RoleX: ['feat0', 'feat1', ...], RoleY: 'TARGET', ....}``.
            **kwargs: Can be used for target/group/weights.

        Returns:
            Dataset with selected features.

        """
        logger.info("\x1b[1mTrain data shape: {}\x1b[0m\n".format(train_data.shape))

        if roles is None:
            roles = {}
        # transform roles from user format {RoleX: ['feat0', 'feat1', ...], RoleY: 'TARGET', ....}
        # to automl format {'feat0': RoleX, 'feat1': RoleX, 'TARGET': RoleY, ...}
        parsed_roles = roles_parser(roles)
        # transform str role definition to automl ColumnRole
        attrs_dict = dict(zip(array_attr_roles, valid_array_attributes))

        for feat in parsed_roles:
            r = parsed_roles[feat]
            if type(r) == str:
                # get default role params if defined
                r = self._get_default_role_from_str(r)

            # check if column is defined like target/group/weight etc ...
            if r.name in attrs_dict:
                # defined in kwargs is rewrited.. TODO: Maybe raise warning if rewrited?
                # TODO: Think, what if multilabel or multitask? Multiple column target ..
                # TODO: Maybe for multilabel/multitask make target only avaliable in kwargs??
                self._used_array_attrs[attrs_dict[r.name]] = feat
                kwargs[attrs_dict[r.name]] = train_data[feat]
                r = DropRole()

            # add new role
            parsed_roles[feat] = r

        assert "target" in kwargs, "Target should be defined"
        self.target = kwargs["target"].name
        kwargs["target"] = self._create_target(kwargs["target"])

        # TODO: Check target and task
        # get subsample if it needed
        subsample = train_data
        if self.samples is not None and self.samples < subsample.shape[0]:
            subsample = subsample.sample(self.samples, axis=0, random_state=42)

        # infer roles
        for feat in subsample.columns:
            assert isinstance(
                feat, str
            ), "Feature names must be string," " find feature name: {}, with type: {}".format(feat, type(feat))
            if feat in parsed_roles:
                r = parsed_roles[feat]
                # handle datetimes
                if r.name == "Datetime":
                    # try if it's ok to infer date with given params
                    try:
                        _ = pd.to_datetime(
                            subsample[feat],
                            format=r.format,
                            origin=r.origin,
                            unit=r.unit,
                        )
                    except ValueError:
                        raise ValueError("Looks like given datetime parsing params are not correctly defined")

                # replace default category dtype for numeric roles dtype if cat col dtype is numeric
                if r.name == "Category":
                    # default category role
                    cat_role = self._get_default_role_from_str("category")
                    # check if role with dtypes was exactly defined
                    try:
                        flg_default_params = feat in roles["category"]
                    except KeyError:
                        flg_default_params = False

                    if (
                        flg_default_params
                        and not np.issubdtype(cat_role.dtype, np.number)
                        and np.issubdtype(subsample.dtypes[feat], np.number)
                    ):
                        r.dtype = self._get_default_role_from_str("numeric").dtype

            else:
                # if no - infer
                if self._is_ok_feature(subsample[feat]):
                    r = self._guess_role(subsample[feat])

                else:
                    r = DropRole()

            # set back
            if r.name != "Drop":
                self._roles[feat] = r
                self._used_features.append(feat)
            else:
                self._dropped_features.append(feat)

        assert len(self.used_features) > 0, "All features are excluded for some reasons"
        # assert len(self.used_array_attrs) > 0, 'At least target should be defined in train dataset'
        # create folds

        folds = set_sklearn_folds(
            self.task,
            kwargs["target"].values,
            cv=self.cv,
            random_state=self.random_state,
            group=None if "group" not in kwargs else kwargs["group"],
        )
        if folds is not None:
            kwargs["folds"] = Series(folds, index=train_data.index)

        # get dataset
        dataset = PandasDataset(train_data[self.used_features], self.roles, task=self.task, **kwargs)
        if self.advanced_roles:
            new_roles = self.advanced_roles_guess(dataset, manual_roles=parsed_roles)

            droplist = [x for x in new_roles if new_roles[x].name == "Drop" and not self._roles[x].force_input]
            self.upd_used_features(remove=droplist)
            self._roles = {x: new_roles[x] for x in new_roles if x not in droplist}
            dataset = PandasDataset(train_data[self.used_features], self.roles, task=self.task, **kwargs)

        return dataset

    def _create_target(self, target: Series):
        """Validate target column and create class mapping is needed

        Args:
            target: Column with target values.

        Returns:
            Transformed target.

        """
        self.class_mapping = None

        if self.task.name != "reg":
            # expect binary or multiclass here
            cnts = target.value_counts(dropna=False)
            assert np.nan not in cnts.index, "Nan in target detected"
            unqiues = cnts.index.values
            srtd = np.sort(unqiues)
            self._n_classes = len(unqiues)

            # case - target correctly defined and no mapping
            if (np.arange(srtd.shape[0]) == srtd).all():

                assert srtd.shape[0] > 1, "Less than 2 unique values in target"
                if self.task.name == "binary":
                    assert srtd.shape[0] == 2, "Binary task and more than 2 values in target"
                return target

            # case - create mapping
            self.class_mapping = {n: x for (x, n) in enumerate(unqiues)}
            return target.map(self.class_mapping).astype(np.int32)

        assert not np.isnan(target.values).any(), "Nan in target detected"
        return target

    def _get_default_role_from_str(self, name) -> RoleType:
        """Get default role for string name according to automl's defaults and user settings.

        Args:
            name: name of role to get.

        Returns:
            role object.

        """
        name = name.lower()
        try:
            role_params = self.roles_params[name]
        except KeyError:
            role_params = {}

        return ColumnRole.from_string(name, **role_params)

    def _guess_role(self, feature: Series) -> RoleType:
        """Try to infer role, simple way.

        If convertable to float -> number.
        Else if convertable to datetime -> datetime.
        Else category.

        Args:
            feature: Column from dataset.

        Returns:
            Feature role.

        """
        # TODO: Plans for advanced roles guessing
        # check if default numeric dtype defined
        num_dtype = self._get_default_role_from_str("numeric").dtype
        # check if feature is number
        try:
            _ = feature.astype(num_dtype)
            return NumericRole(num_dtype)
        except ValueError:
            pass
        except TypeError:
            pass

        # check if default format is defined
        date_format = self._get_default_role_from_str("datetime").format
        # check if it's datetime
        try:
            # TODO: check all notnans and set coerce errors
            _ = cast(
                pd.Series,
                pd.to_datetime(feature, infer_datetime_format=False, format=date_format),
            ).dt.tz_localize("UTC")
            return DatetimeRole(np.datetime64, date_format=date_format)
        except (ValueError, AttributeError):
            # else category
            return CategoryRole(object)

    def _is_ok_feature(self, feature) -> bool:
        """Check if column is filled well to be a feature.

        Args:
            feature: Column from dataset.

        Returns:
            ``True`` if nan ratio and freqency are not high.

        """
        if feature.isnull().mean() >= self.max_nan_rate:
            return False
        if (feature.value_counts().values[0] / feature.shape[0]) >= self.max_constant_rate:
            return False
        return True

    def read(self, data: DataFrame, features_names: Any = None, add_array_attrs: bool = False) -> PandasDataset:
        """Read dataset with fitted metadata.

        Args:
            data: Data.
            features_names: Not used.
            add_array_attrs: Additional attributes, like
              target/group/weights/folds.

        Returns:
            Dataset with new columns.

        """
        kwargs = {}
        if add_array_attrs:
            for array_attr in self.used_array_attrs:
                col_name = self.used_array_attrs[array_attr]
                try:
                    val = data[col_name]
                except KeyError:
                    continue

                if array_attr == "target" and self.class_mapping is not None:
                    val = Series(
                        val.map(self.class_mapping).values,
                        index=data.index,
                        name=col_name,
                    )
                kwargs[array_attr] = val

        dataset = PandasDataset(data[self.used_features], roles=self.roles, task=self.task, **kwargs)

        return dataset

    def advanced_roles_guess(self, dataset: PandasDataset, manual_roles: Optional[RolesDict] = None) -> RolesDict:
        """Advanced roles guess over user's definition and reader's simple guessing.

        Strategy - compute feature's NormalizedGini
        for different encoding ways and calc stats over results.
        Role is inferred by comparing performance stats with manual rules.
        Rule params are params of roles guess in init.
        Defaults are ok in general case.

        Args:
            dataset: Input PandasDataset.
            manual_roles: Dict of user defined roles.

        Returns:
            Dict.

        """
        if manual_roles is None:
            manual_roles = {}
        top_scores = []
        new_roles_dict = dataset.roles
        advanced_roles_params = deepcopy(self.advanced_roles_params)
        drop_co = advanced_roles_params.pop("drop_score_co")
        # guess roles nor numerics

        stat = get_numeric_roles_stat(
            dataset,
            manual_roles=manual_roles,
            random_state=self.random_state,
            subsample=self.samples,
            n_jobs=self.n_jobs,
        )

        if len(stat) > 0:
            # upd stat with rules

            stat = calc_encoding_rules(stat, **advanced_roles_params)
            new_roles_dict = {**new_roles_dict, **rule_based_roles_guess(stat)}
            top_scores.append(stat["max_score"])
        #
        # # # guess categories handling type
        stat = get_category_roles_stat(
            dataset,
            random_state=self.random_state,
            subsample=self.samples,
            n_jobs=self.n_jobs,
        )
        if len(stat) > 0:
            # upd stat with rules
            # TODO: add sample params

            stat = calc_category_rules(stat)
            new_roles_dict = {**new_roles_dict, **rule_based_cat_handler_guess(stat)}
            top_scores.append(stat["max_score"])
        #
        # # get top scores of feature
        if len(top_scores) > 0:
            top_scores = pd.concat(top_scores, axis=0)
            # TODO: Add sample params

            null_scores = get_null_scores(
                dataset,
                list(top_scores.index),
                random_state=self.random_state,
                subsample=self.samples,
            )
            top_scores = pd.concat([null_scores, top_scores], axis=1).max(axis=1)
            rejected = list(top_scores[top_scores < drop_co].index)
            logger.info3("Feats was rejected during automatic roles guess: {0}".format(rejected))
            new_roles_dict = {**new_roles_dict, **{x: DropRole() for x in rejected}}

        return new_roles_dict
