import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import logging 

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


if __name__ == "__main__":
    language_analysis()
