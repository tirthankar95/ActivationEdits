import hydra
from omegaconf import DictConfig
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)
ROOT = 'data/CVEFixes'

# Strategy 1: Randomly split the dataset into train and test sets
def random_split(df: pd.DataFrame, sample_i: int):
    sampled_df = df.sample(frac=0.2)
    sample_dir = Path(ROOT) / 'sample_random'
    sample_dir.mkdir(parents=True, exist_ok=True)
    output_path = sample_dir / f'cve_samples_{sample_i}.csv'
    sampled_df.to_csv(output_path, index=False)
    logger.info(f'Saved {len(sampled_df)} rows to {output_path}')


# Strategy 2: Split the dataset based on programming language
def program_split(df: pd.DataFrame, sample_i: int):
    languages = df['language'].unique()
    for lang in languages:
        lang_df = df[df['language'] == lang]
        sample_dir = Path(ROOT) / f'sample_prog_lang'
        sample_dir.mkdir(parents=True, exist_ok=True)
        output_path = sample_dir / f'{lang}.csv'
        lang_df.to_csv(output_path, index=False)
        logger.info(f'Saved {len(lang_df)} rows for language {lang} to {output_path}')


STRATEGIES = {
    'random': random_split,
    'program': program_split,
}


@hydra.main(version_base=None, config_path="../../config", config_name="lang_split")
def main(cfg: DictConfig):
    df = pd.read_csv(f'{ROOT}/CVEFixes.csv')
    strategy = cfg.strategy
    sample_id = cfg.sample_id

    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {list(STRATEGIES)}")

    logger.info(f"Running strategy='{strategy}' sample_id={sample_id}")
    STRATEGIES[strategy](df, sample_id)


if __name__ == "__main__":
    main()
