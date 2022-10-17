"""Image feaures."""

from typing import Any

import torch

from ...dataset.base import LAMLDataset
from ...image.utils import pil_loader
from ...transformers.base import ColumnsSelector
from ...transformers.base import LAMLTransformer
from ...transformers.base import SequentialTransformer
from ...transformers.base import UnionTransformer
from ...transformers.image import AutoCVWrap
from ...transformers.image import ImageFeaturesTransformer
from ...transformers.numeric import FillInf
from ...transformers.numeric import FillnaMedian
from ...transformers.numeric import StandardScaler
from ..utils import get_columns_by_role
from .base import FeaturesPipeline


class ImageDataFeatures:
    """Class contains basic features transformations for image data."""

    def __init__(self, **kwargs: Any):
        """Set default parameters for image pipeline constructor.

        Args:
            **kwargs: Default parameters.

        """
        self.hist_size = 30
        self.is_hsv = True
        self.n_jobs = 4
        self.loader = pil_loader

        self.embed_model = "efficientnet-b0"
        self.weights_path = None
        self.subs = 10000
        self.cache_dir = "../cache_CV"
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.is_advprop = True
        self.batch_size = 128
        self.verbose = True

        self.random_state = 42

        for k in kwargs:
            self.__dict__[k] = kwargs[k]


class ImageSimpleFeatures(FeaturesPipeline, ImageDataFeatures):
    """Class contains simple color histogram features for image data."""

    def create_pipeline(self, train: LAMLDataset) -> LAMLTransformer:
        transformers_list = []

        # process texts
        imgs = get_columns_by_role(train, "Path")
        if len(imgs) > 0:
            imgs_processing = SequentialTransformer(
                [
                    ColumnsSelector(keys=imgs),
                    ImageFeaturesTransformer(self.hist_size, self.is_hsv, self.n_jobs, self.loader),
                    SequentialTransformer([FillInf(), FillnaMedian(), StandardScaler()]),
                ]
            )
            transformers_list.append(imgs_processing)

        union_all = UnionTransformer(transformers_list)

        return union_all


class ImageAutoFeatures(FeaturesPipeline, ImageDataFeatures):
    """Class contains efficient-net embeddings features for image data."""

    def create_pipeline(self, train: LAMLDataset) -> LAMLTransformer:
        transformers_list = []
        # process texts
        imgs = get_columns_by_role(train, "Path")
        if len(imgs) > 0:
            imgs_processing = SequentialTransformer(
                [
                    ColumnsSelector(keys=imgs),
                    AutoCVWrap(
                        self.embed_model,
                        self.weights_path,
                        self.cache_dir,
                        self.subs,
                        self.device,
                        self.n_jobs,
                        self.random_state,
                        self.is_advprop,
                        self.batch_size,
                        self.verbose,
                    ),
                    SequentialTransformer([FillInf(), FillnaMedian(), StandardScaler()]),
                ]
            )
            transformers_list.append(imgs_processing)

        union_all = UnionTransformer(transformers_list)

        return union_all
