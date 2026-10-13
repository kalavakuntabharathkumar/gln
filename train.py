import argparse
from pathlib import Path
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from gradelens.data import load_answers

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/answers.csv")
    p.add_argument("--output", default="models/gradelens-distilbert")
    p.add_argument("--epochs", type=int, default=2)
    args = p.parse_args()
    df = load_answers(args.data)
    labels = sorted(df.label.unique())
    mapping = {v: i for i, v in enumerate(labels)}
    df["labels"] = df.label.map(mapping)
    train_df, val_df = train_test_split(df, test_size=.2, random_state=42, stratify=df.labels)
    name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(name)
    def tokenize(batch):
        return tokenizer(batch["answer_text"], truncation=True, max_length=256)
    train_ds = Dataset.from_pandas(train_df[["answer_text", "labels"]], preserve_index=False).map(tokenize, batched=True)
    val_ds = Dataset.from_pandas(val_df[["answer_text", "labels"]], preserve_index=False).map(tokenize, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=len(labels))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    def metrics(pair):
        logits, gold = pair
        pred = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(gold, pred),
                "macro_f1": f1_score(gold, pred, average="macro", zero_division=0),
                "quadratic_kappa": cohen_kappa_score(gold, pred, weights="quadratic")}
    mlflow.set_experiment("GradeLens")
    with mlflow.start_run():
        mlflow.log_params({"model": name, "epochs": args.epochs, "train_rows": len(train_df), "validation_rows": len(val_df)})
        training_args = TrainingArguments(output_dir=str(output / "checkpoints"),
            num_train_epochs=args.epochs, per_device_train_batch_size=8,
            per_device_eval_batch_size=16, evaluation_strategy="epoch",
            save_strategy="epoch", logging_steps=20, load_best_model_at_end=True,
            metric_for_best_model="quadratic_kappa", report_to=[])
        trainer = Trainer(model=model, args=training_args, train_dataset=train_ds,
            eval_dataset=val_ds, tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer), compute_metrics=metrics)
        trainer.train()
        result = trainer.evaluate()
        trainer.save_model(str(output))
        tokenizer.save_pretrained(str(output))
        mlflow.log_metrics({k: float(v) for k, v in result.items() if isinstance(v, (int, float))})
        print(result)
        print(f"Saved model to {output}")

if __name__ == "__main__":
    main()
