import pandas as pd
import os

def save_to_csv(csv_path, prompt,gemma_description=None,dot_product=None):
    df=pd.read_csv(csv_path)
    if prompt in df['prompt'].values:
        # Update existing row
        if gemma_description is not None:
            df.loc[df['prompt'] == prompt, 'gemma3 generated description'] = gemma_description
        if dot_product is not None:
            df.loc[df['prompt'] == prompt, 'dot product'] = dot_product

    
csv_path='/Users/baidn/Artificial-Life-and-Foundation-Models/simulation_data/Experiment1_outputs.csv'
