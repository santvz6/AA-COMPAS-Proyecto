import pandas as pd
import numpy as np
import os
import logging

def configure_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_compas_data(input_path: str, output_path: str) -> None:
    """
    Cleans the COMPAS dataset using ProPublica's filtering criteria.
    """
    configure_logging()
    logging.info(f"Loading raw data from {input_path}")
    
    if not os.path.exists(input_path):
        logging.error(f"Input file {input_path} not found.")
        raise FileNotFoundError(f"File {input_path} not found.")
        
    df = pd.read_csv(input_path)
    initial_shape = df.shape
    logging.info(f"Initial shape: {initial_shape}")
    
    # ProPublica filtering
    df = df[
        (df['days_b_screening_arrest'] <= 30) &
        (df['days_b_screening_arrest'] >= -30) &
        (df['is_recid'] != -1) &
        (df['c_charge_degree'] != 'O') &
        (df['score_text'] != 'N/A')
    ].copy()
    
    # We select only the columns that we need for modeling and fairness
    cols_to_keep = [
        'sex', 'age', 'age_cat', 'race', 
        'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count',
        'c_charge_degree', 'c_charge_desc',
        'score_text', 'decile_score', # Target of bias
        'two_year_recid' # Target feature to predict
    ]
    
    # Check if all cols exist
    available_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[available_cols]
    
    # Map 'two_year_recid' to standard target name if needed or just keep it
    
    # Only keep Caucasian and African-American for the bias analysis (as ProPublica did)
    df = df[df['race'].isin(['African-American', 'Caucasian'])].copy()
    
    final_shape = df.shape
    logging.info(f"Final shape after filtering: {final_shape}")
    logging.info(f"Dropped {initial_shape[0] - final_shape[0]} rows.")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"Saved cleaned data to {output_path}")

if __name__ == '__main__':
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    input_file = os.path.join(project_dir, 'data', 'raw', 'compas-scores-two-years.csv')
    output_file = os.path.join(project_dir, 'data', 'processed', 'compas_cleaned.csv')
    clean_compas_data(input_file, output_file)
