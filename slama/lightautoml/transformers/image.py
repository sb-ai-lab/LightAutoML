"""Image features transformers."""

import logging
import os
import pickle

from copy import deepcopy
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import torch

from ..dataset.base import LAMLDataset
from ..dataset.np_pd_dataset import NumpyDataset
from ..dataset.np_pd_dataset import PandasDataset
from ..dataset.roles import NumericRole
from ..image.image import CreateImageFeatures
from ..image.image import DeepImageEmbedder
from ..image.utils import pil_loader
from ..text.utils import get_textarr_hash
from ..text.utils import single_text_hash
from .base import LAMLTransformer


logger = logging.getLogger(__name__)

NumpyOrPandas = Union[NumpyDataset, PandasDataset]


def path_check(dataset: LAMLDataset):
    """Check if all passed vars are path.

    Args:
        dataset: LAMLDataset to check.

    Raises:
         AssertionError: If non-path features are present.

    """
    roles = dataset.roles
    features = dataset.features
    for f in features:
        assert roles[f].name == "Path", "Only path accepted in this transformer"


class ImageFeaturesTransformer(LAMLTransformer):
    """Simple image histogram."""

    _fit_checks = (path_check,)
    _transform_checks = ()
    _fname_prefix = "img_hist"

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

    @property
    def features(self) -> List[str]:
        """Features list.

        Returns:
            List of features names.

        """
        return self._features

    def fit(self, dataset: NumpyOrPandas):
        """Init hist class and create feature names.

        Args:
            dataset: Pandas or Numpy dataset of text features.

        Returns:
            self.

        """
        # set transformer names and add checks
        for check_func in self._fit_checks:
            check_func(dataset)
        # set transformer features

        dataset = dataset.to_pandas()
        df = dataset.data

        feats = []
        self.dicts = {}
        for n, i in enumerate(df.columns):
            fg = CreateImageFeatures(self.hist_size, self.is_hsv, self.n_jobs, self.loader)
            features = list(
                np.char.array([self._fname_prefix + "_"]) + np.char.array(fg.fe.get_names()) + np.char.array(["__" + i])
            )
            self.dicts[i] = {"fg": fg, "feats": features}
            feats.extend(features)
        self._features = feats
        return self

    def transform(self, dataset: NumpyOrPandas) -> NumpyDataset:
        """Transform image dataset to color histograms.

        Args:
            dataset: Pandas or Numpy dataset of image paths.

        Returns:
            Dataset with encoded text.

        """
        # checks here
        super().transform(dataset)
        # convert to accepted dtype and get attributes
        dataset = dataset.to_pandas()
        df = dataset.data

        # transform
        roles = NumericRole()
        outputs = []
        for n, i in enumerate(df.columns):
            new_arr = self.dicts[i]["fg"].transform(df[i].values)
            output = dataset.empty().to_numpy()
            output.set_data(new_arr, self.dicts[i]["feats"], roles)
            outputs.append(output)
        # create resulted
        return dataset.empty().to_numpy().concat(outputs)


class AutoCVWrap(LAMLTransformer):
    """Calculate image embeddings."""

    _fit_checks = (path_check,)
    _transform_checks = ()
    _fname_prefix = "emb_cv"
    _emb_name = ""

    @property
    def features(self) -> List[str]:
        """Features list.

        Returns:
            List of features names.

        """
        return self._features

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
        verbose: bool = True,
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
        self.dicts = {}
        self.cache_dir = cache_dir

        self.transformer = DeepImageEmbedder(
            device,
            n_jobs,
            random_state,
            is_advprop,
            model,
            weights_path,
            batch_size,
            verbose,
        )
        self._emb_name = "DI_" + single_text_hash(self.embed_model)
        self.emb_size = self.transformer.model.feature_shape

    def fit(self, dataset: NumpyOrPandas):
        """Fit chosen transformer and create feature names.

        Args:
            dataset: Pandas or Numpy dataset of text features.

        """
        for check_func in self._fit_checks:
            check_func(dataset)

        if self.cache_dir is not None:
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
        # set transformer features

        # convert to accepted dtype and get attributes
        dataset = dataset.to_pandas()
        df = dataset.data

        # fit
        if self.subs is not None and df.shape[0] >= self.subs:
            subs = df.sample(n=self.subs, random_state=self.random_state)
        else:
            subs = df

        names = []
        for n, i in enumerate(subs.columns):
            feats = [self._fname_prefix + "_" + self._emb_name + "_" + str(x) + "__" + i for x in range(self.emb_size)]
            self.dicts[i] = {
                "transformer": deepcopy(self.transformer.fit(subs[i])),
                "feats": feats,
            }
            names.extend(feats)

        self._features = names
        return self

    def transform(self, dataset: NumpyOrPandas) -> NumpyDataset:
        """Transform dataset to image embeddings.

        Args:
            dataset: Pandas or Numpy dataset of image paths.

        Returns:
            Numpy dataset with image embeddings.

        """
        # checks here
        super().transform(dataset)
        # convert to accepted dtype and get attributes
        dataset = dataset.to_pandas()
        df = dataset.data

        # transform
        roles = NumericRole()
        outputs = []

        for n, conlumn_name in enumerate(df.columns):
            if self.cache_dir is not None:
                full_hash = get_textarr_hash(df[conlumn_name]) + get_textarr_hash(self.dicts[conlumn_name]["feats"])
                fname = os.path.join(self.cache_dir, full_hash + ".pkl")

                if os.path.exists(fname):
                    logger.info3(f"Load saved dataset for {conlumn_name}")

                    with open(fname, "rb") as f:
                        new_arr = pickle.load(f)

                else:
                    new_arr = self.dicts[conlumn_name]["transformer"].transform(df[conlumn_name])
                    with open(fname, "wb") as f:
                        pickle.dump(new_arr, f)
            else:
                new_arr = self.dicts[conlumn_name]["transformer"].transform(df[conlumn_name])

            output = dataset.empty().to_numpy()
            output.set_data(new_arr, self.dicts[conlumn_name]["feats"], roles)
            outputs.append(output)
            logger.info3(f"Feature {conlumn_name} transformed")
        # create resulted
        return dataset.empty().to_numpy().concat(outputs)
