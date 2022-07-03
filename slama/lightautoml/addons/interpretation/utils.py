import itertools

from collections import defaultdict
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from matplotlib.colors import Colormap
from torch.distributions.utils import clamp_probs


T_untokenized = Union[List[str], Tuple[List[str], List[Any]]]


class WrappedVocabulary:
    """Look-up table (word to token-id), handling unrecognized words.

    Args:
        word_to_id: Word to token id dictionary.
        unk_token: Token with which to replace unrecognized words.

    """

    def __init__(self, word_to_id: Dict[str, int], unk_token: str = "<UNK>"):
        self.word_to_id = word_to_id
        self.unk_token = word_to_id[unk_token]

    def __call__(self, x: str) -> int:
        return self.word_to_id.get(x, self.unk_token)

    def __getitem__(self, val: str) -> int:
        return self.word_to_id.get(val, self.unk_token)

    def __len__(self):
        return len(self.word_to_id)


class WrappedTokenizer:
    """Handler for automl tokenizer.

    Args:
        tokenizer: Automl tokenizer.

    """

    def __init__(self, tokenizer: "lightautoml.text.tokenizer.BaseTokenizer"):  # noqa F821
        self._tokenizer = tokenizer

    def __call__(self, x: str) -> List[str]:
        return self._tokenizer.tokenize_sentence(self._tokenizer._tokenize(x))


def untokenize(
    raw: str,
    tokens: List[str],
    return_mask: bool = False,
    token_sym: Any = True,
    untoken_sym: Any = False,
) -> T_untokenized:
    """Get between tokens symbols.

    Args:
        raw: Raw string.
        tokens: List of tokens from raw string.
        return_mask: Flag to return mask
            for each new token. Format: list of
            `token_sym`, `untoken_sym`.
        token_sym: Object, denote token symbol.
        untoken_sym: Object, denote untoken symbol.

    Returns:
        Tuple (full_tokens, tokens_mask) if `return_mask=True`,
            else just list full_tokens.

    """
    mask = []
    untokenized = []
    pos = raw.find(tokens[0])

    if pos != 0:
        untokenized.append(raw[:pos])
        mask.append(untoken_sym)
        raw = raw[pos:]

    prev_token = tokens[0]
    for token in tokens[1:]:
        raw = raw[len(prev_token) :]
        pos = raw.find(token)
        untokenized.append(prev_token)
        mask.append(token_sym)
        if pos:
            mask.append(untoken_sym)
            untokenized.append(raw[:pos])
        prev_token = token
        raw = raw[pos:]

    untokenized.append(prev_token)
    mask.append(token_sym)

    cur = len(prev_token)
    if cur != len(raw):
        untokenized.append(raw[cur:])
        mask.append(untoken_sym)

    if return_mask:
        return untokenized, mask

    return untokenized


def find_positions(arr: List[str], mask: List[bool]) -> List[int]:
    """Set positions and tokens.

    Args:
        tokens: List of tokens and untokens.
        mask: Mask for tokens.

    Returns:
        List of positions of tokens.

    """
    pos = []
    for i, (token, istoken) in enumerate(zip(arr, mask)):
        if istoken:
            pos.append(i)

    return pos


class IndexedString:
    """Indexed string."""

    def __init__(self, raw_string: str, tokenizer: Any, force_order: bool = True):
        """
        Args:
            raw_string: Raw string.
            tokenizer: Tokenizer class.
            force_order: Save order, or use features as
                bag-of-words.

        """
        self.raw = raw_string
        self.tokenizer = tokenizer
        self.force_order = force_order

        self.toks_ = self._tokenize(raw_string)
        self.toks = [token.lower() for token in self.toks_]
        self.as_list_, self.mask = untokenize(self.raw, self.toks_, return_mask=True)

        self.pos = find_positions(self.as_list_, self.mask)
        self.as_np_ = np.array(self.as_list_)
        self.inv = []
        if not force_order:
            pos = defaultdict(list)
            self.vocab = {}
            for token, cur in zip(self.toks, self.pos):
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
                    self.inv.append(token)
                idx = self.vocab[token]
                pos[idx].append(cur)
            self.pos = pos
        else:
            self.inv = self.toks_

    def _tokenize(self, text: str) -> List[str]:
        prep_text = self.tokenizer._tokenize(text)
        tokens = self.tokenizer.tokenize_sentence(prep_text)

        return tokens

    def word(self, idx: int) -> str:
        """Token by its index.

        Args:
            idx: Index of token.

        Returns:
            Token.

        """
        return self.inv[idx]

    def inverse_removing(self, to_del: Union[List[str], List[int]], by_tokens: bool = False) -> str:
        """Remove tokens.

        Args:
            to_del: Tokens (text of int) to del.
            by_tokens: Flag if tokens are text or indexes.

        Returns:
            String without removed tokens.

        """

        # todo: this type of mapping will be not use order,
        # in case when we have not unique tokens.
        assert (not self.force_order) or (self.force_order and not by_tokens)

        if not self.force_order:
            if by_tokens:
                to_del = [self.t_i[token.lower()] for token in to_del]
                to_del = np.array(to_del)
            to_del = list(itertools.chain.from_iterable([self.pos[i] for i in to_del]))
        else:
            to_del = [self.pos[i] for i in to_del]
        mask = np.ones_like(self.as_np_, dtype=bool)
        mask[to_del] = False
        new_str = "".join(self.as_np_[mask])

        return new_str

    @property
    def n_words(self) -> int:
        """Number of unique words."""
        return len(self.pos)


def draw_html(
    tokens_and_weights: List[Tuple[str, float]],
    task_name: str,
    cmap: Any = None,
    grad_line: bool = True,
    grad_positive_label: Optional[str] = None,
    grad_negative_label: Optional[str] = None,
    prediction: Optional[float] = None,
    n_ticks: int = 10,
    draw_order: bool = False,
) -> str:
    """Get colored text in html format.

    For color used gradient from cmap.

    Args:
        tokens_and_weights: List of tokens.
        cmap: ```matplotlib.colors.Colormap``` or single color string like (#FFFFFF).
            By default blue-white-red linear gradient is used.
        positive_label: Positive label text.
        negatvie_label: Negative label text.

    Returns:
        HTML like string.

    """
    font_style = "font-size:14px;"

    token_template = '<span style="background-color: {color_hex};">' "{token}" "</span>"

    ticks_template = (
        '<div style="border-left: 1px solid black; height: 18px; float: left; width: {}%;">'
        '<span style="position: relative; top: 1em; left: -{}em">'
        "{:.1f}"
        "</span>"
        "</div>"
    )

    gradient_full = "margin-left: 10%; " "margin-right: 10%; "

    gradient_styling = (
        "background: linear-gradient(90deg, {} 0%, {} 50%, {} 100%); "
        "border: 1px solid black; "
        "margin-left: {}em; "
        "margin-right: {}em; "
        "float: center; "
    )
    ticks_styling = "margin-left: {}em; " "margin-right: {}em; "
    norm_const = max(map(lambda x: abs(x[1]), tokens_and_weights))
    order = int("{:.2e}".format(norm_const).split("e")[1])
    order_s = "✕ {:.0e}".format(10 ** order)
    scale_word = "Scale"
    if not draw_order:
        order_s = ""
        scale_word = ""
    lord = 0.5 * len(order_s) + 1.5  # lenght order
    inorm_const = 1 / norm_const
    if cmap is None:
        cmap = plt.get_cmap("bwr")

    def get_color_hex(weight):
        if isinstance(cmap, Colormap):
            if task_name == "reg":
                rgba = cmap((-weight * inorm_const * 0.25 + 0.5), bytes=True)
            else:
                rgba = cmap((-weight * inorm_const * 0.25 + 0.5), bytes=True)
                # may be sigmoid for classifications, but hard to understand
                # rgba = cmap(1.0 / (1 + np.exp(weight * inorm_const)), bytes=True)
            return "#%02X%02X%02X" % rgba[:3]
        elif isinstance(cmap, str):
            if weight == 0.0:
                return "#FFFFFF"
            return cmap

    if prediction is None:
        prediction = ""
        pred_field = ""
    else:
        if task_name == "reg":
            prediction = "{:.1e}".format(prediction)
        else:
            prediction = "{:.3f}".format(prediction)
        pred_field = "AutoML's prediction"

    if grad_line:
        if task_name == "reg":
            grad_positive_label = "Positive"
            grad_negative_label = "Negative"
        elif task_name == "multiclass":
            grad_positive_label = "Class: " + grad_positive_label
            grad_negative_label = "Other classes"
        elif task_name == "binary":
            grad_positive_label = "Class: " + grad_positive_label
            grad_negative_label = "Class: " + grad_negative_label
    else:
        grad_positive_label = ""
        grad_negative_label = ""

    tokens_html = [
        token_template.format(token=token, color_hex=get_color_hex(weight)) for token, weight in tokens_and_weights
    ]

    if grad_line:
        between_ticks = [(100 / (n_ticks)) - 5e-2 * 6 / n_ticks if i <= n_ticks - 1 else 0 for i in range(n_ticks + 1)]
        ticks = np.linspace(-norm_const, norm_const, n_ticks + 1) / (10 ** (order))
        ticks_chart = " ".join(
            [ticks_template.format(t, 0.7 + 0.385 * (k < 0), k) for t, k in zip(between_ticks, ticks)]
        )
        grad_statement = """
        <p style="text-align: center">
            Class mapping
        </p>
        <div style="{}">
            <div id="grad" style="{}">
                <p style="text-align:left; margin-left: 1%; margin-right: 1%; color: white;">
                    {}
                    <span style="float:right;">
                        {}
                    </span>
                </p>
            </div>

            <div style="{}">
                {}
            </div>

            <div style="float: right; right: 0.75em; top: -3em; position: relative; font-weight: bold;">{}</div>
            <div style="float: right; right: -2em; top: -2.9em; position: relative; font-weight: bold;">{}</div>

            <div style="float: left; left: -5.5em; top: -4.42em; position: relative;  font-weight: bold;">{}</div>
            <div style="float: left; left: -8.22em; top: -2.9em; position: relative;  font-weight: bold;">{}</div>
        </div>
        """.format(
            gradient_full,
            gradient_styling,
            grad_negative_label,
            grad_positive_label,
            ticks_styling.format(lord, lord),
            ticks_chart,
            scale_word,
            order_s,
            pred_field,
            prediction,
        ).format(
            get_color_hex(-norm_const),
            get_color_hex(0.0),
            get_color_hex(norm_const),
            lord,
            lord,
        )
    else:
        grad_statement = ""

    raw_html = """
    <div>
        <p style="text-align: center">
            Text
        </p>
        <div style="border: 1px solid black;">
            <p style="{}; margin-left: 1%; margin-right: 1%;">{}</p>
        </div>
        {}
    </div>
    """.format(
        font_style, " ".join(tokens_html), grad_statement
    )

    return raw_html


def cross_entropy_multiple_class(input: torch.FloatTensor, target: torch.FloatTensor) -> torch.Tensor:
    return torch.mean(torch.sum(-target * torch.log(clamp_probs(input)), dim=1))
