import tabulate
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy


class MatcherInterface():

    def __init__(self, matcher_method, dataset, column_mapping, target_domain):
        self.matcher_method = matcher_method
        self.dataset = dataset
        self.column_mapping = column_mapping
        self.target_domain = target_domain
        self.mapping_results = None
    
    def calculate_coverage(self):
        if self.mapping_results is None:
            self._match_values()
        sorted_results = sorted(self.mapping_results.items(), key=lambda x: x[1]['coverage'], reverse=True)
        total = 0

        for column_name, match_data in sorted_results:
            coverage = match_data['coverage'] * 100
            total += coverage
            print(f'Column {column_name}: {coverage:.2f}%')
        
        total = total / len(sorted_results)
        print(f'Total: {total:.2f}%')

    def match_values(self, include_unmatches=True):
        if self.mapping_results is None:
            self._match_values()
        
        sorted_results = sorted(self.mapping_results.items(), key=lambda x: x[1]['coverage'], reverse=True)

        for column_name, _ in sorted_results:
            matches = deepcopy(self.mapping_results[column_name]['matches'])
            print(f'Column {column_name}:')

            if include_unmatches:
                for unmatch_value in self.mapping_results[column_name]['unmatch_values']:
                    matches.append((unmatch_value, '-', '-'))
            
            matches_df = pd.DataFrame(matches)
            print(tabulate.tabulate(matches_df, headers=['Current Value', 'Target Value', 'Similarity'], 
                                    tablefmt='orgtbl', showindex=False), '\n')

    def _match_values(self):
        self.mapping_results = {}

        for current_column, target_column in self.column_mapping.items():
            target_values_dict = {x.lower(): x for x in self.target_domain[target_column]}
            current_values_dict = {str(x).strip().lower(): str(x).strip() for x in self.dataset[current_column].unique()}
            self.mapping_results[current_column] = {'matches': None, 'coverage':  None, 
                                                    'unique_values': None, 'unmatch_values': None}
            
            matches_lowercase = self.matcher_method.match(list(current_values_dict.keys()), list(target_values_dict.keys()))
            matches = []

            for current_value, target_value, similarity in matches_lowercase:
                matches.append((current_values_dict[current_value], target_values_dict[target_value], similarity))

            coverage = len(matches) / len(current_values_dict)
            current_values = set(current_values_dict.values())
            match_values = set([x[0] for x in matches])
            self.mapping_results[current_column]['matches'] = matches
            self.mapping_results[current_column]['coverage'] = coverage
            self.mapping_results[current_column]['unique_values'] = current_values
            self.mapping_results[current_column]['unmatch_values'] = current_values - match_values

    def plot_unique_values(self):
        unique_counts = self.dataset.nunique()
        
        # Plot the number of unique values for each column
        plt.figure(figsize=(10, 6))
        unique_counts.plot(kind='bar', color='skyblue')
        plt.xlabel('Columns')
        plt.ylabel('Number of Unique Values')
        plt.xticks(rotation=90)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    from gdc_utils import get_gdc_data
    from table_matching.value_matching_algorithms import TFIDFMatcher, LLMMatcher, EditMatcher

    column_mapping = {
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
    
    dataset = pd.read_csv('../data/use_case1/dou.csv')
    dataset = dataset[column_mapping.keys()]

    gdc_data = get_gdc_data(column_mapping.values())
    matcher = EditMatcher()
    matcher_interface = MatcherInterface(matcher, dataset, column_mapping, gdc_data)
    matcher_interface.calculate_coverage()
    matcher_interface.match_values()
