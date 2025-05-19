import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from dataset import FinancialPhraseBank
from sklearn.model_selection import train_test_split
import pandas as pd

def main():
    model_name = "mistral-small"  # or "gemma-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3
    )

    # PEFT / LoRA setup
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8, lora_alpha=32, lora_dropout=0.05
    )
    model = get_peft_model(base_model, peft_config)

     # Load CSV, then split
    df = pd.read_csv("data/financial_phrasebank.csv")
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    train_ds = FinancialPhraseBank(train_df, tokenizer)
    val_ds   = FinancialPhraseBank(val_df,   tokenizer)

    # Training args
    args = TrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        learning_rate=2e-4,
        evaluation_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        push_to_hub=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model("outputs/finetuned-mistral-lora")
    tokenizer.save_pretrained("outputs/finetuned-mistral-lora")

if __name__ == "__main__":
    main()
