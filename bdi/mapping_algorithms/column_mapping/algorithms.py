from valentine import valentine_match
from valentine.algorithms import SimilarityFlooding

class BaseColumnMappingAlgorithm():
    def __init__(self, dataset, global_table):
        self.dataset = dataset
        self.global_table = global_table

    def map(self):
        raise NotImplementedError("Subclasses must implement this method")

class SimFlood(BaseColumnMappingAlgorithm):

    def __init__(self, dataset, global_table):
        super().__init__(dataset, global_table)
    
    def map(self):
        matcher = SimilarityFlooding()
        matches = valentine_match(self.dataset, self.global_table, matcher)

        mappings = {}
        for match in matches.one_to_one():
            dataset_candidate = match[0][1]
            global_table_candidate = match[1][1]
            mappings[dataset_candidate] = global_table_candidate        
        return mappings


class YurongAlgorithm():

    def __init__(self, dataset, global_table):
        super().__init__(dataset, global_table)
    
    def map(self):
        mappings = {
            #"Proteomics_Participant_ID": "case_submitter_id",
            #"Age": "age_at_diagnosis",
            "Gender": "gender",
            "Race": "race",
            "Ethnicity": "ethnicity",
            #"(none)": "(none)",
            "Histologic_Grade_FIGO": "tumor_grade",
            "tumor_Stage-Pathological": "ajcc_pathologic_stage",
            "Path_Stage_Reg_Lymph_Nodes-pN": "ajcc_pathologic_n",
            "Path_Stage_Primary_Tumor-pT": "ajcc_pathologic_t",
            "Tumor_Focality": "tumor_focality",
            #"Tumor_Size_cm": "tumor_largest_dimension_diameter",
            "Tumor_Site": "tissue_or_organ_of_origin",
            "Histologic_type": "morphology",
            #"Case_excluded": "(none)"
        }
        
        return mappings
