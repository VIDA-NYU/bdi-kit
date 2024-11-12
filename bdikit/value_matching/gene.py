import os
import joblib
import pymart as pm
from typing import List
from bdikit.value_matching.base import BaseValueMatcher, ValueMatch
from bdikit.config import VALUE_MATCHING_THRESHOLD, BDIKIT_CACHE_DIR

BIOMART_CACHE_DIR = os.path.join(BDIKIT_CACHE_DIR, "biomart")
os.makedirs(BIOMART_CACHE_DIR, exist_ok=True)


class Gene(BaseValueMatcher):
    def __init__(
        self,
        source_species: str = "mmusculus",
        target_species: str = "human",
        dataset_name: str = "mmusculus_gene_ensembl",
        top_k: int = 1,
        threshold: float = VALUE_MATCHING_THRESHOLD,
    ):
        self.data = self.__load_data(source_species, target_species, dataset_name)
        self.threshold = threshold
        self.top_k = top_k

    def __load_data(self, source_species, target_species, dataset_name):
        query_cache_path = os.path.join(
            BIOMART_CACHE_DIR,
            f"{source_species}_{target_species}_{dataset_name}.joblib",
        )

        if os.path.exists(query_cache_path):
            return joblib.load(query_cache_path)

        hom_species = [source_species, target_species]
        hom_query = ["ensembl_gene", "orthology_confidence", "perc_id"]

        raw_data = pm.fetch_data(
            dataset_name=dataset_name, hom_species=hom_species, hom_query=hom_query
        )

        raw_data = raw_data[
            [
                "Gene stable ID",
                "Human gene stable ID",
                "%id. target Human gene identical to query gene",
            ]
        ]
        raw_data = raw_data.drop_duplicates()
        raw_data.dropna(inplace=True)

        # Convert the DataFrame to the dictionary format
        data = (
            raw_data.groupby("Gene stable ID")
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

        # Save to cache
        joblib.dump(data, query_cache_path)

        return data

    def match(
        self,
        source_values: List[str],
        target_values: List[str],
    ) -> List[ValueMatch]:
        matches = []
        target_values_set = set(target_values)

        for source_value in source_values:
            match_results = self.data.get(source_value, [])
            for match_result in match_results[: self.top_k]:
                score = match_result["similarity"] / 100
                target = match_result["match"]
                if score >= self.threshold:
                    if target in target_values_set:
                        matches.append(ValueMatch(source_value, target, score))
                else:
                    break

        return matches
