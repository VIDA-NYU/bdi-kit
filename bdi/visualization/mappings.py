import pandas as pd
from copy import deepcopy
from IPython.display import display
from bdi.utils import get_gdc_metadata

pd.set_option('display.max_colwidth', None)

def plot_reduce_scope(reduced_scope, max_chars=150):
    gdc_metadata = get_gdc_metadata()

    for column_data in reduced_scope:
        column_name = column_data['Candidate column']
        recommendations = []
        for candidate_name, candidate_similarity in column_data['Top k columns']:
            candidate_description = gdc_metadata[candidate_name].get('description', '')
            candidate_description = truncate_text(candidate_description, max_chars)
            candidate_values = ', '.join(gdc_metadata[candidate_name].get('enum', []))
            candidate_values = truncate_text(candidate_values, max_chars)
            recommendations.append((candidate_name, candidate_similarity, candidate_description, candidate_values))

        print(f'\n{column_name}:')
        candidates_df = pd.DataFrame(recommendations, columns=['Candidate', 'Similarity', 'Description', 'Values (sample)'])
        display(candidates_df)


def plot_column_mappings(column_mappings):
    column_mappings_df = pd.DataFrame(column_mappings.items(), columns=['Original Column', 'Target Column'])
    display(column_mappings_df)


def plot_value_mappings(value_mappings, include_unmatches=True):
    sorted_results = sorted(value_mappings.items(), key=lambda x: x[1]['coverage'], reverse=True)

    for column_name, _ in sorted_results:
        matches = deepcopy(value_mappings[column_name]['matches'])
        print(f'\nColumn {column_name}:')

        if include_unmatches:
            for unmatch_value in value_mappings[column_name]['unmatch_values']:
                matches.append((unmatch_value, '-', '-'))
        
        matches_df = pd.DataFrame(matches, columns=['Current Value', 'Target Value', 'Similarity'])
        display(matches_df)


def truncate_text(text, max_chars):
    if len(text) > max_chars:
        return text[:max_chars] + '...'
    else:
        return text
