import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import hydra
from pathlib import Path
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from train_low_rank.prepare_data import DatasetPreparer
from peft import LoraConfig, get_peft_model
import logging
import torch

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def fine_tune_lora(
    base_model_name,
    output_adapter_dir="lora_adapter",
    num_train_epochs=64,
    per_device_train_batch_size=4
):
    '''
    per_device_train_batch_size a.k.a batch_size. Device = GPU
    steps_per_epoch = ceil(len(train_dataset) / per_device_train_batch_size)
    total_steps = steps_per_epoch × num_train_epochs
    '''
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    dataset_preparer = DatasetPreparer()
    dataset_preparer.make_train_dataset(tokenizer)
    # LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        use_rslora=True,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)
    # Prepare Data and Train.
    '''
    What the collator does per batch:

    1. Pads dynamically to the longest sequence in the batch
    2. Creates:
    labels = input_ids.clone()
    3. Masks padding tokens in labels as -100
    → ignored by loss

    Important distinction
    mlm=True → BERT-style masked LM ❌
    mlm=False → autoregressive LM ✅
    This is why your dataset didn’t need explicit labels.    
    '''
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=output_adapter_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        logging_steps=4,
        save_strategy="no",  # we will save PEFT adapter separately
        fp16=torch.cuda.is_available(),
        report_to="none"
    )
    for train_dataset in dataset_preparer.load_train_dataset():
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        trainer.train()
    # Save the adapter (PEFT adapter)
    adapter_dir = Path(output_adapter_dir) / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(adapter_dir))
    logging.debug(f"Saved LoRA adapter to {adapter_dir}")
    return str(adapter_dir)


MY_LLM = None
def generate_with_vllm(model_dir, prompt, temperature=0.7, adapter_dir=None):
    global MY_LLM
    if MY_LLM is None:
        llm_kwargs = {
            "model": model_dir,
            "max_model_len": 8192,
            "max_num_batched_tokens": 8192,
            "gpu_memory_utilization": 0.8,
            "dtype": "float16",
        }
        if adapter_dir is not None:
            llm_kwargs["enable_lora"] = True
        MY_LLM = LLM(
            **llm_kwargs
        )
    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature=temperature
    )
    if adapter_dir is not None:
        lora_request = LoRARequest("safe_code_adapter", 1, str(adapter_dir))
        outputs = MY_LLM.generate(
            prompt,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )
    else:
        outputs = MY_LLM.generate(prompt, sampling_params=sampling_params)
    for out in outputs:
        logging.info("=== vllm generation ===")
        logging.info(out.outputs[0].text)
    return outputs


@hydra.main(version_base=None, config_path="../config", config_name="train_low_rank")
def main(cfg: DictConfig):
    logging.debug(OmegaConf.to_yaml(cfg))
    if cfg.train:
        fine_tune_lora(
            base_model_name=cfg.model.name,
            output_adapter_dir=cfg.output.dir,
            num_train_epochs=cfg.training.num_train_epochs,
            per_device_train_batch_size=cfg.training.per_device_train_batch_size,
            merge_and_save=cfg.output.merge_and_save,
        )
    elif cfg.inference.use_adapter:
            adapter_dir = Path(cfg.inference.adapter_dir)
            if not adapter_dir.exists():
                raise FileNotFoundError(f"No LoRA adapter found at {adapter_dir}")
            logging.info(
                f"Loading base model {cfg.model.name} with adapter {adapter_dir} and generating for prompt:\n{cfg.generation.query}\n"
            )
            # sanity check
            generate_with_vllm(
                cfg.model.name,
                "Q: Summarize: Python is simple.\nA:",
                temperature=0.7,
                adapter_dir=str(adapter_dir),
            )
    else:
        logging.warning('Mode not recognized. Please set either train or inference.use_adapter to true in the config.')


def get_module_names(model_dir: str):
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    for name, module in model.named_modules():
        print(name, module.__class__.__name__)


if __name__ == "__main__":
    main()