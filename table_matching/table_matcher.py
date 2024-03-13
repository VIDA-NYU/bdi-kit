import os
import pandas as pd
from valentine.metrics import F1Score, PrecisionTopNPercent
from valentine import valentine_match
from valentine.algorithms import JaccardDistanceMatcher
from valentine.algorithms import Coma
from valentine.algorithms import SimilarityFlooding
from valentine.algorithms import DistributionBased
from valentine.algorithms import Cupid
import pprint

pp = pprint.PrettyPrinter(indent=4, sort_dicts=True)


TARGET_TABLE_NAME = 'target' #TODO move to a constant definition file
CANDIDATE_TABLE_NAME = 'candidate'

def detect_matching_columns(candidate_df, target_df, groundtruth=None, matcher=None):
    """
    Detect matching columns between two tables.
    Args:
        candidate_df (pd.DataFrame): The discovered table.
        target_df (pd.DataFrame): The target table.
        groundtruth (list): The groundtruth list of tuples, where each tuple represents a mapping from candidate_df to target.
        config (dict): A dictionary with the configuration for the matcher.
    Returns:
        dict: A dictionary with the matching columns.
    """
    # TODO add more matchers, and configs to the matchers
    if matcher is None:
        # matcher = JaccardDistanceMatcher()
        # matcher = Cupid()
        # matcher = SimilarityFlooding()
        # matcher = DistributionBased()
        matcher = Coma()

    print(f"Computing the matches using the {type(matcher).__name__} algorithm...")

    # matches = valentine_match(candidate_df, target_df, matcher, CANDIDATE_TABLE_NAME, TARGET_TABLE_NAME)
    matches = valentine_match(candidate_df, target_df, matcher)

    # MatcherResults is a wrapper object that has several useful
    # utility/transformation functions
    print("Found the following matches:")
    pp.pprint(matches)

    print("\nOne-to-one matches:")
    pp.pprint(matches.one_to_one())

    non_covered = target_df.columns.difference(matches.one_to_one().values())
    print(f"\nColumns not covered: {non_covered}")

    print("\nThe MatcherResults object is a dict and can be treated such:")
    for match in matches:
        print(f"{str(match): <60} {matches[match]}")

    if groundtruth is not None:

        metrics = matches.get_metrics(groundtruth)

        print("\nAccording to the ground truth:")
        pp.pprint(groundtruth)

        print("\nThese are the scores of the default metrics for the matcher:")
        pp.pprint(metrics)

        print("\nYou can also get specific metric scores:")
        pp.pprint(matches.get_metrics(groundtruth, metrics={
            PrecisionTopNPercent(n=80),
            F1Score()
        }))

    return matches


