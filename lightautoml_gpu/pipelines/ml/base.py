"""Base classes for MLPipeline."""

from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

from lightautoml_gpu.validation.base import TrainValidIterator

import torch
if torch.cuda.is_available():
    from ...dataset.gpu.gpu_dataset import CupyDataset, CudfDataset, DaskCudfDataset
else:
    print("could not load gpu related libs (pipelines/ml/base.py)")

from ...dataset.np_pd_dataset import NumpyDataset, PandasDataset

from ...dataset.base import LAMLDataset
from ...dataset.utils import concatenate
from ...ml_algo.base import MLAlgo
from ...ml_algo.tuning.base import DefaultTuner
from ...ml_algo.tuning.base import ParamsTuner
from ...ml_algo.utils import tune_and_fit_predict
from ..features.base import EmptyFeaturePipeline
from ..features.base import FeaturesPipeline
from ..selection.base import EmptySelector
from ..selection.base import SelectionPipeline


class MLPipeline:
    """Single ML pipeline.

    Merge together stage of building ML model
    (every step, excluding model training, is optional):

        - Pre selection: select features from input data.
          Performed by
          :class:`~lightautoml_gpu.pipelines.features.SelectionPipeline`.
        - Features generation: build new features from selected.
          Performed by
          :class:`~lightautoml_gpu.pipelines.features.FeaturesPipeline`.
        - Post selection: One more selection step - from created features.
          Performed by
          :class:`~lightautoml_gpu.pipelines.features.SelectionPipeline`.
        - Hyperparams optimization for one or multiple ML models.
          Performed by
          :class:`~lightautoml_gpu.ml_algo.tuning.base.ParamsTuner`.
        - Train one or multiple ML models:
          Performed by :class:`~lightautoml_gpu.ml_algo.base.MLAlgo`.
          This step is the only required for at least 1 model.

    Args:
        ml_algos: Sequence of MLAlgo's or Pair - (MlAlgo, ParamsTuner).
        force_calc: Flag if single fold of ml_algo
            should be calculated anyway.
        pre_selection: Initial feature selection.
            If ``None`` there is no initial selection.
        features_pipeline: Composition of feature transforms.
        post_selection: Post feature selection.
            If ``None`` there is no post selection.

    """

    @property
    def used_features(self) -> List[str]:  # noqa D102
        return self.pre_selection.selected_features

    def __init__(
        self,
        ml_algos: Sequence[Union[MLAlgo, Tuple[MLAlgo, ParamsTuner]]],
        force_calc: Union[bool, Sequence[bool]] = True,
        pre_selection: Optional[SelectionPipeline] = None,
        features_pipeline: Optional[FeaturesPipeline] = None,
        post_selection: Optional[SelectionPipeline] = None,
    ):
        if pre_selection is None:
            pre_selection = EmptySelector()

        self.pre_selection = pre_selection

        if features_pipeline is None:
            features_pipeline = EmptyFeaturePipeline()

        self.features_pipeline = features_pipeline

        if post_selection is None:
            post_selection = EmptySelector()

        self.post_selection = post_selection

        self._ml_algos = []
        self.params_tuners = []

        for n, mt_pair in enumerate(ml_algos):

            try:
                # case when model and tuner are defined
                mod, tuner = mt_pair
            except (TypeError, ValueError):
                # case when only model is definded
                mod, tuner = mt_pair, DefaultTuner()

            mod.set_prefix("Mod_{0}".format(n))

            self._ml_algos.append(mod)
            self.params_tuners.append(tuner)

        self.force_calc = [force_calc] * len(self._ml_algos) if type(force_calc) is bool else force_calc
        # TODO: Do we need this assert?
        assert any(self.force_calc), "At least single algo in pipe should be forced to calc"

    def to_cpu(self):
        """Base method for pipeline conversion to CPU inference mode.

        Returns:
            instance of pipeline with CPU inference.
        """
        def convert_recursive_cpu(pipeline):
            if hasattr(pipeline, 'transformer_list'):
                for i in range(len(pipeline.transformer_list)):
                    convert_recursive_cpu(pipeline.transformer_list[i])
            else:
                if hasattr(pipeline, 'to_cpu'):
                    pipeline.to_cpu()
                if hasattr(pipeline, 'dataset_type'):
                    if pipeline.dataset_type == DaskCudfDataset or \
                            pipeline.dataset_type == CudfDataset:
                        pipeline.dataset_type = PandasDataset
                    elif pipeline.dataset_type == CupyDataset:
                        pipeline.dataset_type = NumpyDataset

        convert_recursive_cpu(self.features_pipeline._pipeline)

        for i in range(len(self.ml_algos)):
            self.ml_algos[i] = self.ml_algos[i].to_cpu()
        return self

    def fit_predict(self, train_valid: TrainValidIterator) -> LAMLDataset:
        """Fit on train/valid iterator and transform on validation part.

        Args:
            train_valid: Dataset iterator.

        Returns:
            Dataset with predictions of all models.

        """
        self.ml_algos = []
        # train and apply pre selection
        train_valid = train_valid.apply_selector(self.pre_selection)

        # apply features pipeline
        train_valid = train_valid.apply_feature_pipeline(self.features_pipeline)

        # train and apply post selection
        train_valid = train_valid.apply_selector(self.post_selection)

        predictions = []
        for ml_algo, param_tuner, force_calc in zip(self._ml_algos, self.params_tuners, self.force_calc):
            ml_algo, preds = tune_and_fit_predict(ml_algo, param_tuner, train_valid, force_calc)
            if ml_algo is not None:
                self.ml_algos.append(ml_algo)

                predictions.append(preds)

        assert (
            len(predictions) > 0
        ), "Pipeline finished with 0 models for some reason.\nProbably one or more models failed"

        predictions = concatenate(predictions)

        del self._ml_algos
        return predictions

    def predict(self, dataset: LAMLDataset) -> LAMLDataset:
        """Predict on new dataset.

        Args:
            dataset: Dataset used for prediction.

        Returns:
            Dataset with predictions of all trained models.

        """
        dataset = self.pre_selection.select(dataset)
        dataset = self.features_pipeline.transform(dataset)
        dataset = self.post_selection.select(dataset)

        predictions = []

        for model in self.ml_algos:
            pred = model.predict(dataset)
            predictions.append(pred)

        predictions = concatenate(predictions)

        return predictions

    def upd_model_names(self, prefix: str):
        """Update prefix pipeline models names.

        Used to fit inside AutoML where multiple models
        with same names may be trained.

        Args:
            prefix: New prefix name.

        """
        for n, mod in enumerate(self._ml_algos):
            mod.set_prefix(prefix)

    def prune_algos(self, idx: Sequence[int]):
        """Prune model from pipeline.

        Used to fit blender - some models may be excluded from final ensemble.

        Args:
            idx: Selected algos.

        """
        self.ml_algos = [x for (n, x) in enumerate(self.ml_algos) if n not in idx]
