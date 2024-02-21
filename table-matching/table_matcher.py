import os
import pandas as pd
from valentine.metrics import F1Score, PrecisionTopNPercent
from valentine import valentine_match
from valentine.algorithms import JaccardDistanceMatcher
import pprint
pp = pprint.PrettyPrinter(indent=4, sort_dicts=True)

EXTRACTED_TABLES_DIRECTORY = "data/extracted-tables"
TARGET_TABLE = "data/target.csv"

TARGET_TABLE_NAME = 'target'
DISCOVERED_TABLE_NAME = 'discovered'

SELECTED_COLS_THRESHOLD = 0.2


def main():
    target_df = pd.read_csv(TARGET_TABLE)
    print(f'Looking for matching columns for:')  
    print(f'Columns: {target_df.columns}')
    pp.pprint(target_df)
    print('\n\n')

    selected_pairs = {}
    for filename in sorted(os.listdir(EXTRACTED_TABLES_DIRECTORY)):
        table_name  = filename.replace('.csv', '')
        full_filename = os.path.join(EXTRACTED_TABLES_DIRECTORY, filename)
        if os.path.isfile(full_filename) and filename.endswith(".csv"):
            print(f'Looking for matching columns in {filename} ...')
            df = pd.read_csv(full_filename)
            
            matcher = JaccardDistanceMatcher()

            result = valentine_match(
                target_df, df, matcher, TARGET_TABLE_NAME, table_name)

            for pair, score in result.items():
                if score >= SELECTED_COLS_THRESHOLD:
                    selected_pairs[pair] = score

                    col_target = target_df[pair[0][1]]
                    col_discovered = df[pair[1][1]]
                    
                    combined_df = pd.concat([col_target, col_discovered], axis=1)
                    print(combined_df)


            break

    pp.pprint(selected_pairs)


if __name__ == '__main__':
    main()
