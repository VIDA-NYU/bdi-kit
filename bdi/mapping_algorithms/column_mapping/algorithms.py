class YurongAlgorithm():

    def __init__(self, *args):
        pass
    
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
