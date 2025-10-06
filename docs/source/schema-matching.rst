Schema Matching Methods
=======================

This page provides an overview of all schema matching methods available in the `bdikit` library.
Some methods reuse the implementation of other libraries such as `Valentine <https://delftdata.github.io/valentine/>`_ (e.g, `similarity_flooding`, `coma` and `cupid`) while others are implemented originally for bdikit (e.g., `gpt`, `ct_learning`, and `two_phase`).
To see how to use these methods, please refer to the documentation of :py:func:`~bdikit.api.match_schema()` in the :py:mod:`~bdikit.api` module.

.. ``bdikit module <api>`.



.. list-table:: bdikit methods
    :header-rows: 1
    
    * - Method
      - Class
      - Description
    * - ``magneto_zs_bp``
      - :class:`~bdikit.schema_matching.magneto.MagnetoZSBP`
      - | Uses a zero-shot small language model as retriever with the bipartite algorithm as reranker in Magneto.
    * - ``magneto_ft_bp``
      - :class:`~bdikit.schema_matching.magneto.MagnetoFTBP`
      - | Uses a fine-tuned small language model as retriever with the bipartite algorithm as reranker in Magneto.
    * - ``magneto_zs_llm``
      - :class:`~bdikit.schema_matching.magneto.MagnetoZSLLM`
      - | Uses a zero-shot small language model as retriever with a large language model as reranker in Magneto.
    * - ``magneto_ft_llm``
      - :class:`~bdikit.schema_matching.magneto.MagnetoFTLLM`
      - | Uses a fine-tuned small language model as retriever with a large language model as reranker in Magneto.
    * - ``max_val_sim``
      - :class:`~bdikit.schema_matching.twophase.MaxValSim`
      - | This schema matching method first uses a a top-k column matcher (e.g., `ct_learning`) to prune the search space (keeping only the top-k most likely matches), and then uses a value matcher method to choose the best match from the pruned search space.
    * - ``two_phase``
      - :class:`~bdikit.schema_matching.twophase.TwoPhase`
      - | The two-phase schema matching method first uses a a top-k column matcher (e.g., `ct_learning`) to prune the search space (keeping only the top-k most likely matches), and then uses another column matcher to choose the best match from the pruned search space.
    * - ``llm``
      - :class:`~bdikit.schema_matching.llm.LLM`
      - | Leverages LLMs to identify and select the most accurate schema matches. Supports multiple models, with `gpt-4o-mini` used as the default.

.. list-table:: Methods from other libraries
    :header-rows: 1
    
    * - Method
      - Class
      - Description
    * - ``similarity_flooding``
      - :class:`~bdikit.schema_matching.valentine.SimFlood`
      - | Similarity Flooding transforms schemas into directed graphs and merges them into a propagation graph. The algorithm iteratively propagates similarity scores to neighboring nodes until convergence. This algorithm was proposed by Sergey Melnik, Hector Garcia-Molina, and Erhard Rahm in "Similarity Flooding: A Versatile Graph Matching Algorithm and Its Application to Schema Matching" (ICDE, 2002).
    * - ``coma``
      - :class:`~bdikit.schema_matching.valentine.Coma`
      - | COMA is a matcher that combines multiple schema-based matchers, representing schemas as rooted directed acyclic graphs. This algorithm was proposed by Do, Hong-Hai, and Erhard Rahm in "COMA — a system for flexible combination of schema matching approaches." (VLDB 2002). *This algorithm requires Java to be installed on the system.*
    * - ``cupid``
      - :class:`~bdikit.schema_matching.valentine.Cupid`
      - | Cupid is a schema-based approach that translates schemas into tree structures. It calculates overall similarity using linguistic and structural similarities, with tree transformations helping to compute context-based similarity. This algorithm was proposed by Madhavan et al. in "Generic Schema Matching with Cupid" (VLDB, 2001)​.
    * - ``distribution_based``
      - :class:`~bdikit.schema_matching.valentine.DistributionBased`
      - | Distribution-based Matching compares the distribution of data values in columns using the Earth Mover's Distance. It clusters relational attributes based on these comparisons. This algorithm was proposed by Zhang et al. in "Automatic discovery of attributes in relational databases" (SIGMOD 2011).
    * - ``jaccard_distance``
      - :class:`~bdikit.schema_matching.valentine.Jaccard`
      - | This algorithm computes pairwise column similarities using Jaccard similarity, treating values as identical if their Levenshtein distance is below a threshold. The algorithm was proposed by Koutras et al. in "Valentine: Evaluating matching techniques for dataset discovery" (ICDE 2021).
