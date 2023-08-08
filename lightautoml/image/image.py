"""Image feature extractors based on color histograms and CNN embeddings."""

from copy import copy
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union


try:
    import cv2
except:
    import warnings

    warnings.warn("'cv2' - package isn't installed")

import numpy as np
import torch
import torch.nn as nn


try:
    import timm

    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
except:
    import warnings

    warnings.warn("'timm' - package isn't installed")

from joblib import Parallel
from joblib import delayed
from sklearn.base import TransformerMixin
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..text.utils import parse_devices
from ..text.utils import seed_everything
from .utils import pil_loader


numeric = Union[int, float]


class ColorFeatures:
    """Basic class to compute color features."""

    def __init__(self, hist_size: int = 30, is_hsv: bool = True):
        """Create normalized color histogram for rgb or hsv image.

        Works with RGB and grayscale images only.

        Args:
            hist_size: Number of bins for each channel.
            is_hsv: Convert image to hsv.

        """
        self.hist_size = hist_size
        self.is_hsv = is_hsv
        self._f_names = ["h", "s", "v"] if self.is_hsv else ["r", "g", "b"]

    def compute_histogram(self, img: np.ndarray) -> List[numeric]:
        """Compute normalized color histogram for one channel.

        Args:
            img: Image with shape ``(h, w)``.

        Returns:
            List of channel histogram values.

        """
        # TODO: add value range check
        hist = cv2.calcHist([img], [0], mask=None, histSize=[self.hist_size], ranges=(0, 255))[:, 0]

        return list(hist / hist.sum())

    def get_names(self) -> List[str]:
        """Define how to get features names list.

        Returns:
            List of features names.

        """
        return [
            j
            for i in [["color_" + j + "_" + str(i) for i in np.arange(self.hist_size)] for j in self._f_names]
            for j in i
        ]

    def get_features(self, img: np.ndarray) -> List[numeric]:
        """Calculate normalized color histogram for rgb or hsv image.

        Args:
            img: Image with shape ``(h, w)``.

        Returns:
            List of histogram values.

        """
        img = copy(img)
        if self.is_hsv:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        res = []
        for i in range(3):
            channel = img[:, :, i]
            res.extend(self.compute_histogram(channel))

        return res


class CreateImageFeatures:
    """Class for parallel histogram computation."""

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
        self.fe = ColorFeatures(hist_size, is_hsv)
        self.n_jobs = n_jobs
        self.loader = loader

    def process(self, im_path_i: str) -> List[numeric]:
        """Create normalized color histogram for input image by its path.

        Args:
            im_path_i: Path to the image.

        Returns:
            List of histogram values.

        """
        im_i = np.array(self.loader(im_path_i))
        if len(im_i.shape) == 2:
            im_i = cv2.cvtColor(im_i, cv2.COLOR_GRAY2RGB)
        res_i = self.fe.get_features(im_i)
        return res_i

    def transform(self, samples: Sequence[str]) -> np.ndarray:
        """Transform input sequence with paths to histogram values.

        Args:
            samples: Sequence with images paths.

        Returns:
            Array of histograms.

        """
        res = Parallel(self.n_jobs)(delayed(self.process)(im_path_i) for im_path_i in samples)
        return np.vstack(res)


class TimmModelEmbedder(nn.Module):
    """Class to compute TimmModels embeddings."""

    def __init__(
        self,
        model_name: str = "efficientnet_b0.ra_in1k",
        weights_path: Optional[str] = None,
        device=torch.device("cuda:0"),
    ):
        """Pytorch module for image embeddings based on timm models.

        Args:
            model_name: Name of effnet model.
            weights_path: Path to saved weights.
            device: Device to use.

        """
        super(TimmModelEmbedder, self).__init__()
        self.device = device
        self.model = (
            timm.create_model(model_name, pretrained=True, num_classes=0, checkpoint_path=weights_path)
            .eval()
            .to(self.device)
        )
        self.feature_shape = self.get_shape()
        self.model_name = model_name

    @torch.no_grad()
    def get_shape(self) -> int:
        """Calculate output embedding shape.

        Returns:
            Shape of embedding.

        """
        return self.model(torch.randn(1, 3, 224, 224).to(self.device)).squeeze().shape[0]

    def forward(self, x) -> torch.Tensor:
        """Forward pass."""
        out = self.model(x)
        return out


class ImageTimmDataset:
    """Image for Timm Dataset Class."""

    def __init__(
        self,
        model: TimmModelEmbedder,
        data: Sequence[str],
        loader: Callable = pil_loader,
    ):
        """Pytorch Dataset for :class:`~lightautoml.image.TimmModelEmbedder`.

        Args:
            model: model which we train.
            data: Sequence of paths.
            loader: Callable for reading image from path.

        """
        self.X = data
        self.transforms = create_transform(**resolve_data_config(model.model.pretrained_cfg, model=model.model))
        self.loader = loader

    def __getitem__(self, idx: int) -> np.ndarray:
        path = self.X[idx]
        img = self.loader(path)
        img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.X)


class DeepTimmImageEmbedder(TransformerMixin):
    """Timm Transformer for image embeddings."""

    def __init__(
        self,
        device: torch.device = torch.device("cuda:0"),
        n_jobs=4,
        random_state=42,
        model_name="efficientnet_b0.ra_in1k",
        weights_path: Optional[str] = None,
        batch_size: int = 128,
        verbose: bool = True,
    ):
        """Pytorch Dataset for :class:`~lightautoml.image.TimmModelEmbedder`.

        Args:
            device: Torch device.
            n_jobs: Number of threads for dataloader.
            random_state: Random seed.
            model_name: Name of effnet model.
            weights_path: Path to saved weights.
            batch_size: Batch size.
            verbose: Verbose data processing.

        """
        super(DeepTimmImageEmbedder, self).__init__()
        # add assert to check model
        # assert model_name in {f"efficientnet-b{i}" for i in range(8)}

        self.device, self.device_ids = parse_devices(device)
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.verbose = verbose
        seed_everything(random_state)

        self.model = TimmModelEmbedder(model_name, weights_path, self.device)

    def fit(self, data: Any = None):
        """Train model."""
        return self

    @torch.no_grad()
    def transform(self, data: Sequence[str]) -> np.ndarray:
        """Calculate image embeddings from pathes.

        Args:
            data: Sequence of paths.

        Returns:
            Array of embeddings.

        """
        data = ImageTimmDataset(self.model, data)
        loader = DataLoader(data, batch_size=self.batch_size, shuffle=False, num_workers=self.n_jobs)

        result = []
        if self.verbose:
            loader = tqdm(loader)

        self.model = self.model.to(self.device)
        for batch in loader:
            embed = self.model(batch.float().to(self.device)).detach().cpu().numpy()
            result.append(embed.astype(np.float32))

        return np.vstack(result)
