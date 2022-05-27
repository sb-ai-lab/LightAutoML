from array import ArrayType
from copy import deepcopy
from typing import List, Optional, Dict, Tuple, Callable, Sequence, TypeVar, Iterator

import torch
import numpy as np
import pandas as pd
from pyspark import Broadcast
from pyspark.sql.pandas.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import FloatType

from lightautoml.dataset.base import LAMLDataset
from lightautoml.image.image import DeepImageEmbedder, CreateImageFeatures
from lightautoml.image.utils import pil_loader
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.dataset.roles import NumericVectorOrArrayRole
from lightautoml.spark.transformers.base import ObsoleteSparkTransformer
from lightautoml.text.utils import single_text_hash
from lightautoml.transformers.image import path_check


def vector_or_array_check(dataset: LAMLDataset):
    roles = dataset.roles
    features = dataset.features
    for f in features:
        assert isinstance(roles[f], NumericVectorOrArrayRole), "Only NumericVectorRole is accepted"


def path_or_vector_check(dataset: LAMLDataset):
    try:
        path_check(dataset)
        return
    except AssertionError:
        pass

    try:
        vector_or_array_check(dataset)
        return
    except AssertionError:
        pass

    assert False, "All incoming features should have same roles of either Path or NumericVector"


# TODO: needs a vector based alternative
class ImageFeaturesTransformer(ObsoleteSparkTransformer):
    """Simple image histogram."""

    _fit_checks = (path_check,)
    _transform_checks = ()
    _fname_prefix = "img_hist"

    _can_unwind_parents = False

    def __init__(
        self,
        hist_size: int = 30,
        is_hsv: bool = True,
        n_jobs: int = 4,
        loader: Callable = pil_loader,
    ):
        """Create normalized color histogram for rgb or hsv image.

        Args:
            hist_size: Number of bins for each channel.
            is_hsv: Convert image to hsv.
            n_jobs: Number of threads for multiprocessing.
            loader: Callable for reading image from path.

        """
        self.hist_size = hist_size
        self.is_hsv = is_hsv
        self.n_jobs = n_jobs
        self.loader = loader
        self._fg: Optional[CreateImageFeatures] = None

    @property
    def features(self) -> List[str]:
        """Features list.

        Returns:
            List of features names.

        """
        return self._features

    def _fit(self, dataset: SparkDataset):
        """Init hist class and create feature names.

        Args:
            dataset: Pandas or Numpy dataset of text features.

        Returns:
            self.

        """

        sdf = dataset.data
        self._fg = CreateImageFeatures(self.hist_size, self.is_hsv, self.n_jobs, self.loader)
        self._features = [f"{self._fname_prefix}__{c}" for c in sdf.columns]

        return self

    def _transform(self, dataset: SparkDataset) -> SparkDataset:
        """Transform image dataset to color histograms.

        Args:
            dataset: Pandas or Numpy dataset of image paths.

        Returns:
            Dataset with encoded text.

        """

        sdf = dataset.data

        # transform
        roles = []
        new_cols = []
        for c, out_col_name in zip(dataset.features, self.features):
            role = NumericVectorOrArrayRole(
                size=len(self._fg.fe.get_names()),
                element_col_name_template=[f"{self._fname_prefix}_{name}__{c}" for name in self._fg.fe.get_names()],
                dtype=np.float32,
                is_vector=False
            )
            fg_bcast = dataset.spark_session.sparkContext.broadcast(self._fg)

            @pandas_udf("array<float>", PandasUDFType.SCALAR)
            def calculate_embeddings(data: pd.Series) -> pd.Series:
                fg = fg_bcast.value
                img_embeds = pd.Series(list(fg.transform(data)))
                return img_embeds

            new_cols.append(calculate_embeddings(c).alias(out_col_name))
            roles.append(role)

        new_sdf = sdf.select(*dataset.service_columns, *new_cols)

        output = dataset.empty()
        output.set_data(new_sdf, self.features, roles)

        return output


class AutoCVWrap(ObsoleteSparkTransformer):
    """Calculate image embeddings."""
    _fit_checks = ()
    _transform_checks = ()
    _fname_prefix = "emb_cv"
    _emb_name = ""

    _T = TypeVar('_T')

    @property
    def features(self) -> List[str]:
        """Features list.

        Returns:
            List of features names.

        """
        return self._features

    @property
    def _image_loader(self):
        raise NotImplementedError()

    def __init__(
            self,
            model="efficientnet-b0",
            weights_path: Optional[str] = None,
            cache_dir: str = "./cache_CV",
            subs: Optional = None,
            device: torch.device = torch.device("cuda:0"),
            n_jobs: int = 4,
            random_state: int = 42,
            is_advprop: bool = True,
            batch_size: int = 128,
            verbose: bool = True
    ):
        """

        Args:
            model: Name of effnet model.
            weights_path: Path to saved weights.
            cache_dir: Path to cache directory or None.
            subs: Subsample to fit transformer. If ``None`` - full data.
            device: Torch device.
            n_jobs: Number of threads for dataloader.
            random_state: Random state to take subsample and set torch seed.
            is_advprop: Use adversarial training.
            batch_size: Batch size for embedding model.
            verbose: Verbose data processing.

        """
        self.embed_model = model
        self.random_state = random_state
        self.subs = subs
        self.cache_dir = cache_dir
        self._img_transformers: Optional[Dict[str, Tuple[DeepImageEmbedder, str]]] = None

        self.transformer = DeepImageEmbedder(
            device,
            n_jobs,
            random_state,
            is_advprop,
            model,
            weights_path,
            batch_size,
            verbose,
            image_loader=self._image_loader
        )

        self._emb_name = "DI_" + single_text_hash(self.embed_model)
        self.emb_size = self.transformer.model.feature_shape

    def _fit(self, dataset: SparkDataset):
        """Fit chosen transformer and create feature names.

        Args:
            dataset: Pandas or Numpy dataset of text features.

        """

        sdf = dataset.data

        self._img_transformers = dict()
        for c in dataset.features:
            out_column_name = f"{self._fname_prefix}_{self._emb_name}__{c}"

            self._img_transformers[c] = (
                # TODO: we don't really want 'fit' here, because it would happen on the driver side
                # TODO: better to mark fitless classes with some Marker type via inheritance
                # TODO: to avoid errors of applying the wrong transformer as early as possible
                deepcopy(self.transformer.fit(sdf.select(c))),
                out_column_name
            )

        self._features = [feat for _, feat in self._img_transformers.values()]
        return self

    def _transform(self, dataset: SparkDataset) -> SparkDataset:
        """Transform dataset to image embeddings.

        Args:
            dataset: Pandas or Numpy dataset of image paths.

        Returns:
            Numpy dataset with image embeddings.

        """

        sdf = dataset.data

        # transform
        roles = []
        new_cols = []
        for c in dataset.features:
            role = NumericVectorOrArrayRole(
                size=self.emb_size,
                element_col_name_template=f"{self._fname_prefix}_{self._emb_name}_{{}}__{c}",
                dtype=np.float32,
                is_vector=False
            )

            # TODO: probably transformer should be created on the worker side and not in the driver
            trans, out_col_name = self._img_transformers[c]
            transformer_bcast = dataset.spark_session.sparkContext.broadcast(value=trans)

            @pandas_udf("array<float>", PandasUDFType.SCALAR)
            def calculate_embeddings(data: pd.Series) -> pd.Series:
                transformer = transformer_bcast.value
                img_embeds = pd.Series(list(transformer.transform(data)))
                return img_embeds

            new_cols.append(calculate_embeddings(c).alias(out_col_name))
            roles.append(role)

        new_sdf = sdf.select(*dataset.service_columns, *new_cols)

        output = dataset.empty()
        output.set_data(new_sdf, self.features, roles)

        return output


class PathBasedAutoCVWrap(AutoCVWrap):
    _fit_checks = (path_check,)
    _T = str

    def __init__(self, image_loader: Callable, *args, **kwargs):
        self.__image_loader = image_loader
        super().__init__(*args, **kwargs)

    @property
    def _image_loader(self):
        return self.__image_loader


class ArrayBasedAutoCVWrap(AutoCVWrap):
    _fit_checks = (vector_or_array_check,)
    _T = bytes

    def __init__(self, *args, **kwargs):
        self.__image_loader = lambda x: x
        super().__init__(*args, **kwargs)

    @property
    def _image_loader(self):
        return self.__image_loader
