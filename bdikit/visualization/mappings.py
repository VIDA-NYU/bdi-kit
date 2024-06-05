import pandas as pd
from copy import deepcopy
from IPython.display import display
from bdikit.visualization.scope_reducing import ScopeReducerExplorer
from bdikit.visualization.scope_reducing import SRHeatMapManager

pd.set_option("display.max_colwidth", None)


def plot_reduce_scope(reduced_scope, dataset):
    scope_explorer = SRHeatMapManager(dataset, reduced_scope)
    scope_explorer.get_heatmap()
    display(scope_explorer.plot_heatmap())


def plot_column_mappings(column_mappings):
    column_mappings_df = pd.DataFrame(
        column_mappings.items(), columns=["Original Column", "Target Column"]
    )
    display(column_mappings_df)


def plot_value_mappings(value_mappings, include_unmatches=True):
    sorted_results = sorted(
        value_mappings.items(), key=lambda x: x[1]["coverage"], reverse=True
    )

    for column_name, _ in sorted_results:
        matches = deepcopy(value_mappings[column_name]["matches"])
        print(f"\nColumn {column_name}:")

        if include_unmatches:
            for unmatch_value in value_mappings[column_name]["unmatch_values"]:
                matches.append((unmatch_value, "-", "-"))

        matches_df = pd.DataFrame(
            matches, columns=["Current Value", "Target Value", "Similarity"]
        )
        display(matches_df)
