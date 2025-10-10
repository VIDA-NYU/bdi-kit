Value Matching Methods
======================

This page provides an overview of all value matching methods available in the `bdikit` library.
Some methods reuse the implementation of other libraries such as `PolyFuzz <https://maartengr.github.io/PolyFuzz/>`_ (e.g, `embedding` and `tfidf`) while others are implemented originally for bdikit (e.g., `gpt`).
To see how to use these methods, please refer to the documentation of :py:func:`~bdikit.api.match_values()` in the :py:mod:`~bdikit.api` module.

.. ``bdikit module <api>`.



.. list-table:: bdikit methods
    :header-rows: 1
    
    * - Method
      - Class
      - Description
    * - ``llm``
      - :class:`~bdikit.value_matching.llm.LLM`
      - | Leverages LLMs to identify and select the most accurate value matches. Supports multiple models, with `gpt-4o-mini` used as the default.
    * - ``llm_numeric``
      - :class:`~bdikit.value_matching.llm_numeric.LLMNumeric`
      - | Employs LLMs to perform numeric value transformations, such as converting ages from years to months. Supports multiple models, with `gpt-4o-mini` used as the default.

.. list-table:: Methods from other libraries
    :header-rows: 1
    
    * - Method
      - Class
      - Description
    * - ``tfidf``
      - :class:`~bdikit.value_matching.polyfuzzs.TFIDF`
      - | Employs a character-based n-gram TF-IDF approach to approximate edit distance by capturing the frequency and contextual importance of n-gram patterns within strings. This method leverages the Term Frequency-Inverse Document Frequency (TF-IDF) weighting to quantify the similarity between strings based on their shared n-gram features.
    * - ``edit_distance``
      - :class:`~bdikit.value_matching.polyfuzz.EditDistance`
      - | Uses the edit distance between lists of strings using a customizable scorer that supports various distance and similarity metrics.
    * - ``embedding``
      - :class:`~bdikit.value_matching.polyfuzz.Embeddings`
      - | A value-matching algorithm that leverages the cosine similarity of value embeddings for precise comparisons. By default, it utilizes the `bert-base-multilingual-cased` model to generate contextualized embeddings, enabling effective multilingual matching.​