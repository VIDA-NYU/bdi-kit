import pandas as pd
import matplotlib.pyplot as plt
from bdi.mapping_algorithms.value_mapping.algorithms import TFIDFMatcher, LLMMatcher, EditMatcher


class ValueMappingManager():

    def __init__(self, dataset, column_mapping, target_domain):
        self.matcher_method = EditMatcher()
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

    def skip_values(self, values):
        max_length = 50
        if isinstance(values[0], float):
            return True
        elif len(values) > max_length:
            return True
        else:
            return False


    def map(self):
        if self.mapping_results is None:
            self._match_values()

        return self.mapping_results


    def _match_values(self):
        self.mapping_results = {}

        for current_column, target_column in self.column_mapping.items():
            if self.target_domain[target_column] is None:
                continue
            target_values_dict = {x.lower(): x for x in self.target_domain[target_column]}
            unique_values = self.dataset[current_column].unique()

            if self.skip_values(unique_values):
                continue
            current_values_dict = {str(x).strip().lower(): str(x).strip() for x in unique_values}
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
