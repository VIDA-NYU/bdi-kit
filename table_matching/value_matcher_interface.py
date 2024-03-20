import tabulate
import pandas as pd
import matplotlib.pyplot as plt


class MatcherInterface():

    def __init__(self, matcher_method, dataset, column_mapping, target_domain):
        self.matcher_method = matcher_method
        self.dataset = dataset
        self.column_mapping = column_mapping
        self.target_domain = target_domain
        self.mapping_results = None
    
    def calculate_coverage(self):
        if self.mapping_results is None:
            self._match_values(verbose=False)
        sorted_results = sorted(self.mapping_results.items(), key=lambda x: x[1]['coverage'], reverse=True)
        total = 0

        for column_name, match_data in sorted_results:
            coverage = match_data['coverage'] * 100
            total += coverage
            print(f'Column {column_name}: {coverage:.2f}%')
        
        total = total / len(sorted_results)
        print(f'Total: {total:.2f}%')

    def match_values(self):
        self._match_values(verbose=True)

    def _match_values(self, verbose=True):
        self.mapping_results = {}

        for current_column, target_column in self.column_mapping.items():
            target_values = self.target_domain[target_column]
            current_values = list(set([str(x).strip() for x in self.dataset[current_column].unique()]))
            self.mapping_results[current_column] = {'matches': None, 'coverage':  None}
            matches = []
            if target_values is not None:
                matches = self.matcher_method.match(current_values, target_values)
            
            coverage = len(matches) / len(current_values)
            self.mapping_results[current_column]['matches'] = matches
            self.mapping_results[current_column]['coverage'] = coverage
            if verbose:
                print(f'Column {current_column}:')
                matches_df = pd.DataFrame(matches)
                if len(matches_df) > 0:
                    print(tabulate.tabulate(matches_df, headers=['Current Value', 'Target Value', 'Similarity'], 
                                            tablefmt='orgtbl', showindex=False), '\n')

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
    from value_matcher import TFIDFMatcher

    column_mapping = {
        #"Proteomics_Participant_ID": "case_submitter_id",
        "Age": "age_at_diagnosis",
        "Gender": "gender",
        "Race": "race",
        "Ethnicity": "ethnicity",
        #"(none)": "(none)",
        "Histologic_Grade_FIGO": "tumor_grade",
        "tumor_Stage-Pathological": "ajcc_pathologic_stage",
        "Path_Stage_Reg_Lymph_Nodes-pN": "ajcc_pathologic_n",
        "Path_Stage_Primary_Tumor-pT": "ajcc_pathologic_t",
        "Tumor_Focality": "tumor_focality",
        "Tumor_Size_cm": "tumor_largest_dimension_diameter",
        "Tumor_Site": "tissue_or_organ_of_origin",
        "Histologic_type": "morphology",
        #"Case_excluded": "(none)"
    }
    
    dataset = pd.read_csv('../data/use_case1/dou.csv')
    dataset = dataset[column_mapping.keys()]

    gdc_data = get_gdc_data(column_mapping.values())
    matcher = TFIDFMatcher()
    matcher_interface = MatcherInterface(matcher, dataset, column_mapping, gdc_data)
    matcher_interface.calculate_coverage()
    matcher_interface.match_values()
