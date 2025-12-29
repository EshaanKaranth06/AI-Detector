import os
import json
import torch
import numpy as np
import re
import logging
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding,
    EarlyStoppingCallback
)
from transformers.trainer_utils import PredictionOutput
from common_utils import convert_doc_to_pdf, ocr_pdf, unstructured_to_markdown, is_text_pdf
import gc
from pathlib import Path
from typing import Tuple, List, Dict, Any, cast

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class OptimizedKaggleAIResumeDetector:
    def __init__(self, model_name="microsoft/deberta-v3-small"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2,
            torch_dtype=torch.float32
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            torch.cuda.empty_cache()

    def load_kaggle_datasets(self, data_dir="./kaggle_data", max_samples_per_class=421) -> Tuple[List[str], List[int]]:
        human_texts, ai_texts = [], []
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

        for file in csv_files:
            df = pd.read_csv(os.path.join(data_dir, file), dtype=str)
            is_ai_dataset = 'ai' in file.lower() or 'generated' in file.lower()

            texts = []
            for _, row in df.iterrows():
                combined_text = " ".join(str(cell).strip() for cell in row if pd.notna(cell))
                if 100 <= len(combined_text) <= 2000:
                    texts.append(combined_text)

            if is_ai_dataset:
                ai_texts.extend(texts)
            else:
                human_texts.extend(texts)

        min_count = min(len(human_texts), len(ai_texts), max_samples_per_class)
        logger.info(f"Using {min_count} Human + {min_count} AI resumes for balanced dataset")

        np.random.seed(42)
        human_sample = np.random.choice(human_texts, min_count, replace=False)
        ai_sample = np.random.choice(ai_texts, min_count, replace=False)

        texts = list(human_sample) + list(ai_sample)
        labels = [1] * min_count + [0] * min_count
        combined = list(zip(texts, labels))
        np.random.shuffle(combined)

        texts_out: List[str]
        labels_out: List[int]
        texts_out, labels_out = zip(*combined)  # type: ignore
        return list(texts_out), list(labels_out)

    def prepare_dataset(self, texts: List[str], labels: List[int]) -> Any:
        """Prepare dataset for training. Returns Dataset object."""
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"], 
                truncation=True, 
                padding=False,
                max_length=384
            )
        dataset = Dataset.from_dict({"text": texts, "labels": labels})
        return dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    def compute_metrics(self, eval_pred):
        preds, labels = eval_pred
        preds = np.argmax(preds, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        accuracy = accuracy_score(labels, preds)
        return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

    def create_training_args(self, output_dir: str) -> TrainingArguments:
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=24,
            gradient_accumulation_steps=3,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            fp16=False,
            bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
            dataloader_pin_memory=True,
            dataloader_num_workers=2,
            gradient_checkpointing=True,
            weight_decay=0.01,
            max_grad_norm=1.0,
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            report_to="none",
            remove_unused_columns=True,
            optim="adamw_torch",
        )

    def train(self, output_dir: str, data_dir: str):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        texts, labels = self.load_kaggle_datasets(data_dir=data_dir)
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, labels, test_size=0.15, stratify=labels, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.15, stratify=y_temp, random_state=42
        )
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        train_ds = self.prepare_dataset(X_train, y_train)
        val_ds = self.prepare_dataset(X_val, y_val)
        test_ds = self.prepare_dataset(X_test, y_test)
        
        del texts, labels, X_temp, y_temp
        gc.collect()

        args = self.create_training_args(output_dir)

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8),
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        logger.info("Starting training...")
        trainer.train()

        logger.info("Evaluating on test set...")
        # The Dataset typing is complex in HF, so we use Any to avoid type errors
        predictions = trainer.predict(test_ds)
        pred_labels = np.argmax(predictions.predictions, axis=1)

        print("\\nFinal Test Set Results:")
        print(classification_report(
            y_test, pred_labels,
            target_names=["AI Generated", "Human Written"],
            digits=4
        ))

        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return trainer
    
    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = ''.join(char for char in text if char.isprintable() or char.isspace())
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_text(self, file_path: Path) -> str:
        try:
            file_path = Path(file_path)
            suffix = file_path.suffix.lower()

            if suffix == ".txt":
                try:
                    with file_path.open("r", encoding="utf-8") as f:
                        return self.clean_text(f.read())
                except UnicodeDecodeError:
                    with file_path.open("r", encoding="latin-1") as f:
                        return self.clean_text(f.read())

            delete_pdf = False

            if suffix in [".doc", ".docx"]:
                pdf_path = convert_doc_to_pdf(file_path)
                delete_pdf = True
                if not pdf_path or not pdf_path.exists():
                    logger.warning(f"Failed to convert DOC/DOCX: {file_path}")
                    return ""
            else:
                pdf_path = file_path

            text = ""
            if is_text_pdf(pdf_path):
                text = unstructured_to_markdown(pdf_path)
            if not text.strip():
                text = ocr_pdf(pdf_path)

            if delete_pdf and pdf_path.exists():
                try:
                    pdf_path.unlink()
                    logger.info(f"Deleted intermediate PDF: {pdf_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete temp PDF: {pdf_path} -> {e}")

            return self.clean_text(text)

        except Exception as e:
            logger.error(f"Text extraction failed for {file_path}: {e}")
            return ""

    def predict(self, text: str, model_path: str) -> Dict[str, Any]:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        model.eval()
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=384
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze()
        
        # Convert to int for indexing
        prediction_idx = int(torch.argmax(probs).item())
        
        return {
            "prediction": "Human Written" if prediction_idx == 1 else "AI Generated",
            "confidence": float(probs[prediction_idx].item()),
            "ai_probability": float(probs[0].item()),
            "human_probability": float(probs[1].item())
        }

    def predict_folder(self, input_dir: str, output_dir: str, model_path: str):
        input_dir_path = Path(input_dir)
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        all_files = [f for f in input_dir_path.iterdir() if f.suffix.lower() in ['.txt', '.pdf', '.doc', '.docx']]
        logger.info(f"Found {len(all_files)} resumes in '{input_dir_path}'")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        model.eval()

        if torch.cuda.is_available():
            model = model.cuda()

        for resume_file in all_files:
            output_file = output_dir_path / (resume_file.stem + ".json")
            if output_file.exists():
                logger.info(f"Skipping already processed file: {resume_file.name}")
                continue

            text = self.extract_text(resume_file)

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=384
            )
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).squeeze()

            # Convert to int for indexing
            prediction_idx = int(torch.argmax(probs).item())
            
            result = {
                "file": resume_file.name,
                "prediction": "Human Written" if prediction_idx == 1 else "AI Generated",
                "confidence": float(probs[prediction_idx].item()),
                "ai_probability": float(probs[0].item()),
                "human_probability": float(probs[1].item())
            }

            with output_file.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=4)

            logger.info(f"Processed: {resume_file.name} -> {result['prediction']}")

def get_all_dataset_files(data_dir: str) -> List[str]:
    return sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])

def save_metadata(path: str, dataset_files: List[str]):
    with open(path, "w") as f:
        json.dump({"datasets": dataset_files}, f)

def load_metadata(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

def main():
    model_name = "microsoft/deberta-v3-small"
    data_dir = "./kaggle_data"
    output_dir = "./ai_resume_detector_optimized"
    os.makedirs(output_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        logger.info(f"CUDA available. GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        logger.info("BF16 supported" if torch.cuda.is_bf16_supported() else "BF16 not supported")

    torch.set_num_threads(6)

    mode = input("Enter mode ('train' or 'infer'): ").strip().lower()

    detector = OptimizedKaggleAIResumeDetector(model_name=model_name)

    if mode == "train":
        current_files = get_all_dataset_files(data_dir)
        metadata_path = os.path.join(output_dir, "train_metadata.json")
        metadata = load_metadata(metadata_path)

        if metadata and metadata.get("datasets") == current_files:
            logger.info("Dataset unchanged. Skipping training.")
        else:
            trainer = detector.train(output_dir, data_dir)
            save_metadata(metadata_path, current_files)

    elif mode == "infer":
        test_dir = input("Enter folder path with resumes (.pdf, .doc, .docx, .txt): ").strip()
        out_file = "results/ai_results.json"

        test_dir_path = Path(test_dir)
        out_file_path = Path(out_file)

        if not test_dir_path.exists():
            logger.error(f"Invalid input folder: {test_dir_path}")
            return

        all_files = [f for f in test_dir_path.iterdir() if f.suffix.lower() in ['.txt', '.pdf', '.doc', '.docx']]
        logger.info(f"Found {len(all_files)} resumes in '{test_dir_path}'")

        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            output_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()

        results = []
        for resume_file in all_files:
            logger.info(f"Processing: {resume_file.name}")
            text = detector.extract_text(resume_file)
            if not text.strip():
                logger.warning(f"Empty or unreadable text in file: {resume_file.name}")
                continue

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=384
            )
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).squeeze()

            # Convert to int for indexing
            prediction_idx = int(torch.argmax(probs).item())
            
            result = {
                "file": resume_file.name,
                "prediction": "Human Written" if prediction_idx == 1 else "AI Generated",
                "confidence": float(probs[prediction_idx].item()),
                "ai_probability": float(probs[0].item()),
                "human_probability": float(probs[1].item())
            }

            results.append(result)
            logger.info(f"{resume_file.name} → {result['prediction']} ({result['confidence']:.4f})")

        # Ensure output directory exists
        out_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with out_file_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        logger.info(f"\nAll predictions saved to: {out_file_path}")

    else:
        logger.error("Invalid mode. Please enter 'train' or 'infer'.")

if __name__ == "__main__":
    main()