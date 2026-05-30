import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import logging 
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)
ROOT = 'data/CVEFixes'
df = pd.read_csv(f'{ROOT}/CVEFixes.csv') 

def language_analysis():
    logger.info(f'Data shape: {df.shape}')
    df['code_length'] = df['code'].apply(lambda x: len(str(x).split(' ')))
    logger.info(f'Code length statistics: {df["code_length"].describe()}')
    logger.info(f'Programming Languages: {df["language"].value_counts()}')
    language_counts = df['language'].value_counts().head(20)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=language_counts.index, y=language_counts.values)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'{ROOT}/language_distribution.png')


def language_sample(sample_i: int):
    sampled_df = df.sample(frac=0.2)
    sample_dir = Path(ROOT) / f'sample'
    sample_dir.mkdir(parents=True, exist_ok=True)
    output_path = sample_dir / f'cve_samples_{sample_i}.csv'
    sampled_df.to_csv(output_path, index=False)
    logger.info(f'Saved {len(sampled_df)} rows to {output_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Language analytics and sampling utility')
    parser.add_argument('--si', type=int, help='Sample folder suffix for data/sample_i')
    args = parser.parse_args()
    language_analysis()
    if args.si is not None:
        language_sample(args.si)
