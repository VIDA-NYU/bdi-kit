from typing import List
import pymart as pm
from bdikit.value_matching.base import BaseValueMatcher, ValueMatch
from bdikit.config import VALUE_MATCHING_THRESHOLD


class Gene(BaseValueMatcher):
    def __init__(
        self,
        source_species: str = "mmusculus",
        target_species: str = "human",
        dataset_name: str = "mmusculus_gene_ensembl",
        top_k: int = 1,
        threshold: float = VALUE_MATCHING_THRESHOLD,
    ):
        hom_species = [source_species, target_species]
        hom_query = ["ensembl_gene", "orthology_confidence", "perc_id"]

        data = pm.fetch_data(
            dataset_name=dataset_name, hom_species=hom_species, hom_query=hom_query
        )

        data = data[
            [
                "Gene stable ID",
                "Human gene stable ID",
                "%id. target Human gene identical to query gene",
            ]
        ]
        data = data.drop_duplicates()
        data.dropna(inplace=True)

        # Convert the DataFrame to the dictionary format
        self.matches = (
            data.groupby("Gene stable ID")
            .apply(
                lambda group: sorted(
                    [
                        {
                            "match": row["Human gene stable ID"],
                            "similarity": row[
                                "%id. target Human gene identical to query gene"
                            ],
                        }
                        for _, row in group.iterrows()
                    ],
                    key=lambda x: x["similarity"],  # Sort by similarity
                    reverse=True,
                )
            )
            .to_dict()
        )
        self.threshold = threshold
        self.top_k = top_k

    def match(
        self,
        source_values: List[str],
        target_values: List[str],
    ) -> List[ValueMatch]:
        matches = []
        target_values_set = set(target_values)

        for source_value in source_values:
            match_results = self.matches.get(source_value, [])
            print(source_value, match_results)
            for match_result in match_results[: self.top_k]:
                score = match_result["similarity"] / 100
                target = match_result["match"]
                if score >= self.threshold:
                    if target in target_values_set:
                        matches.append(ValueMatch(source_value, target, score))
                else:
                    break

        return matches
