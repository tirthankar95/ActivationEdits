from datasets import load_dataset

ds = load_dataset("stasvinokur/cve-and-cwe-dataset-1999-2025")
df = ds['train'].to_pandas()
df = df.groupby('CWE-ID').first().reset_index()
df = df[['CWE-ID', 'DESCRIPTION']]
df.to_csv('data/CVEFixes/CWE_descriptions.csv', index=False)