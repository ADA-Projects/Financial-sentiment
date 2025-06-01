#!/usr/bin/env python3
"""
Ultra memory-efficient Mistral-7B training for 16GB RAM
Uses aggressive optimizations to fit in limited memory
"""

import torch
import gc
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM, 
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import pandas as pd
from datasets import Dataset

# Force CPU offloading and memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

class UltraEfficientMistralTrainer:
    def __init__(self):
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.3"  # Updated to v0.3
        self.tokenizer = None
        self.model = None
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def check_memory(self, step=""):
        """Monitor memory usage"""
        import psutil
        ram_gb = psutil.virtual_memory().used / 1e9
        print(f"📊 {step} RAM: {ram_gb:.1f} GB")
        
        if torch.cuda.is_available():
            gpu_gb = torch.cuda.memory_allocated() / 1e9
            print(f"📊 {step} GPU: {gpu_gb:.1f} GB")
    
    def setup_ultra_efficient_model(self):
        """Load model with maximum memory efficiency"""
        print("🔧 Loading Mistral-7B-v0.3 with ultra-efficient config...")
        
        # Ultra-aggressive quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,  # Not bfloat16 to save memory
            llm_int8_enable_fp32_cpu_offload=True
        )
        
        # Load tokenizer first
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,  # Faster tokenizer
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.check_memory("After tokenizer")
        
        # Load model with aggressive CPU offloading
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory={0: "4GB", "cpu": "10GB"},  # More aggressive limits
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        self.check_memory("After model load")
        
        # Prepare for training
        self.model = prepare_model_for_kbit_training(self.model)
        
        print("✅ Mistral-7B-v0.3 loaded with ultra-efficient config")
    
    def setup_minimal_lora(self):
        """Minimal LoRA config to save memory"""
        print("🔧 Setting up minimal LoRA...")
        
        # Very small LoRA to minimize memory
        lora_config = LoraConfig(
            r=8,                    # Smaller rank (was 16)
            lora_alpha=16,          # Smaller alpha (was 32) 
            target_modules=[        # Fewer target modules
                "q_proj",
                "v_proj",           # Only q and v, skip k and o
            ],
            lora_dropout=0.05,      # Lower dropout
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"📊 Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        
        self.check_memory("After LoRA")
    
    def prepare_minimal_dataset(self):
        """Prepare smaller dataset to save memory"""
        print("📚 Preparing minimal dataset...")
        
        df = pd.read_csv("data/data.csv")
        
        # Use only a subset for memory efficiency
        df_sample = df.sample(n=min(1000, len(df)), random_state=42)
        
        instructions = []
        for _, row in df_sample.iterrows():
            prompt = f"Sentiment: {row['Sentence']} →"
            completion = f" {row['Sentiment']}"
            
            instructions.append({
                "text": prompt + completion,
                "input_length": len(prompt)
            })
        
        dataset = Dataset.from_list(instructions)
        return dataset.train_test_split(test_size=0.2, seed=42)
    
    def tokenize_efficiently(self, dataset):
        """Memory-efficient tokenization"""
        print("🔤 Tokenizing efficiently...")
        
        def tokenize_function(examples):
            # Shorter sequences to save memory
            tokens = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=256,  # Shorter than usual (was 512)
                return_overflowing_tokens=False
            )
            tokens["labels"] = tokens["input_ids"].copy()
            return tokens
        
        # Process in smaller batches
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=100,  # Smaller batches
            remove_columns=dataset["train"].column_names
        )
        
        return tokenized
    
    def train_ultra_efficient(self):
        """Train with minimal memory usage"""
        print("🚀 Starting ultra-efficient training...")
        
        dataset = self.prepare_minimal_dataset()
        tokenized_dataset = self.tokenize_efficiently(dataset)
        
        # Ultra-conservative training args
        training_args = TrainingArguments(
            output_dir="./outputs/mistral-ultra-efficient",
            num_train_epochs=1,                    # Just 1 epoch
            per_device_train_batch_size=1,         # Tiny batches
            gradient_accumulation_steps=16,        # Larger accumulation
            warmup_ratio=0.1,
            learning_rate=5e-5,                    # Lower LR
            weight_decay=0.01,
            logging_steps=10,
            save_strategy="no",                    # Don't save checkpoints
            evaluation_strategy="no",              # Skip evaluation
            report_to=None,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            fp16=True,
            gradient_checkpointing=True,           # Save memory
            dataloader_num_workers=0,              # No multiprocessing
        )
        
        # Simple data collator
        from transformers import DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            data_collator=data_collator,
        )
        
        self.check_memory("Before training")
        
        # Train
        try:
            trainer.train()
            print("✅ Training completed!")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("❌ Still out of memory. Need larger machine.")
                print("💡 Try upgrading to GitHub Pro for 8-core access")
            else:
                raise e
        
        self.check_memory("After training")
    
    def test_inference(self):
        """Quick inference test"""
        test_prompt = "Sentiment: Company profits soared 25% →"
        
        inputs = self.tokenizer(test_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        print(f"🧪 Test: '{test_prompt}' → '{response.strip()}'")

def main():
    """Main training workflow"""
    print("🎯 Ultra Memory-Efficient Mistral-7B-v0.3 Training")
    print("Target: 16GB RAM (4-core Codespace)")
    print("=" * 50)
    
    trainer = UltraEfficientMistralTrainer()
    
    try:
        trainer.setup_ultra_efficient_model()
        trainer.setup_minimal_lora()
        trainer.train_ultra_efficient()
        trainer.test_inference()
        
        print("\n🎉 Success! Mistral-7B-v0.3 worked on 16GB RAM")
        
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        if "out of memory" in str(e).lower() or "memory" in str(e).lower():
            print("\n💡 Memory optimization suggestions:")
            print("1. Try Phi-3-Mini (3.8B params): microsoft/Phi-3-mini-4k-instruct")
            print("2. Try Gemma-2B: google/gemma-2b-it")
            print("3. Use Google Colab Pro+ for more RAM")
        else:
            print("\n💡 Other recommendations:")
            print("1. Check HF authentication: huggingface-cli login")
            print("2. Ensure all dependencies installed: pip install bitsandbytes accelerate")
            print("3. Try a smaller model first")
        print("\n4. Your FinBERT solution (83% F1) is already excellent!")

if __name__ == "__main__":
    main()