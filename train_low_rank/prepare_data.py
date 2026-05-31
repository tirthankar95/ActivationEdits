from datasets import Dataset, concatenate_datasets, load_from_disk
from pathlib import Path
import pandas as pd
import logging
import shutil

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)    
DATA_DIR = Path('data/CVEFixes/sample')
DATA_SAVE = Path('data/train')

class DatasetPreparer:
    def __init__(self):
        pass


    def make_train_dataset(self, tokenizer):
        if DATA_SAVE.exists():
            logger.info(f"Training data-folder exists: {DATA_SAVE}")
            return 

        def tokenize_fn(examples):
            # max_length: 512, doesn't mean padding is added to make the length 512
            tokens = tokenizer(examples["text"], truncation=True, max_length=512)
            return tokens

        def render_messages_as_text(messages):
            # Prefer model-specific chat formatting when available.
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            # Fallback for tokenizers without a chat template.
            return "\n".join([f"{m['role']}: {m['content']}" for m in messages])

        indx = 0
        for filename in DATA_DIR.glob("*.csv"):
            df = pd.read_csv(filename)
            df['language'] = df['language'].apply(lambda x: 'any' if x == 'Other' else x)
            samples = []
            for _, row in df.iterrows():
                messages = [
                    {
                        "role": "user",
                        "content": (
                            "You are a helpful coding assistant. "
                            f"Generate a code snippet in {row['language']} programming language "
                            f"which is {row['safety']}."
                        ),
                    },
                    {"role": "assistant", "content": str(row["code"])},
                ]
                samples.append({"text": render_messages_as_text(messages)})
            file_ds = Dataset.from_list(samples)
            file_ds = file_ds.map(tokenize_fn, remove_columns=["text"])
            if not file_ds:
                raise ValueError(f"No CSV files found in {DATA_DIR}")
            file_ds.set_format(type="torch")
            
            # Save the processed dataset and load it back from DATA_SAVE.
            DATA_SAVE.parent.mkdir(parents=True, exist_ok=True)
            file_ds.save_to_disk(str(DATA_SAVE / f"sample_{indx}"))
            indx += 1
        logger.info(f"Finished preparing training datasets. Datasets saved to {DATA_SAVE}.")


    def load_train_dataset(self):
        for ds_file in DATA_SAVE.glob("sample_*"):
            if ds_file.is_file():
                logger.info(f"Loading dataset from {ds_file}")
                ds = load_from_disk(str(ds_file))
                ds.set_format(type="torch")
                yield ds
            else:
                logger.warning(f"Error loading file: {ds_file}")

