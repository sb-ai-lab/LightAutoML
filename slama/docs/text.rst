.. role:: hidden
    :class: hidden-section


lightautoml.text
==============================

Provides an internal interface for working with text features.

Sentence Embedders
------------------------------

.. currentmodule:: lightautoml.text

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    ~dl_transformers.DLTransformer
    ~dl_transformers.BOREP
    ~dl_transformers.RandomLSTM
    ~dl_transformers.BertEmbedder
    ~weighted_average_transformer.WeightedAverageTransformer


Torch Datasets for Text
------------------------------

.. currentmodule:: lightautoml.text

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    ~embed_dataset.BertDataset
    ~embed_dataset.EmbedDataset


Tokenizers
------------------------------

.. currentmodule:: lightautoml.text

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    ~tokenizer.BaseTokenizer
    ~tokenizer.SimpleRuTokenizer
    ~tokenizer.SimpleEnTokenizer


Pooling Strategies
------------------------------

.. currentmodule:: lightautoml.text

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: classtemplate.rst

    ~sentence_pooling.SequenceAbstractPooler
    ~sentence_pooling.SequenceClsPooler
    ~sentence_pooling.SequenceMaxPooler
    ~sentence_pooling.SequenceSumPooler
    ~sentence_pooling.SequenceAvgPooler
    ~sentence_pooling.SequenceIndentityPooler


Utils
------------------------------

.. currentmodule:: lightautoml.text

.. autosummary::
    :toctree: ./generated
    :nosignatures:
    :template: functiontemplate.rst

    ~utils.seed_everything
    ~utils.parse_devices
    ~utils.custom_collate
    ~utils.single_text_hash
    ~utils.get_textarr_hash
