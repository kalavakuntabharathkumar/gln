import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from gradelens.data import load_answers

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/answers.csv")
    p.add_argument("--model", default="models/gradelens-distilbert")
    args = p.parse_args()
    df = load_answers(args.data)
    labels = sorted(df.label.unique())
    df["labels"] = df.label.map({v:i for i,v in enumerate(labels)})
    _, test = train_test_split(df, test_size=.2, random_state=42, stratify=df.labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    ds = Dataset.from_pandas(test[["answer_text", "labels"]], preserve_index=False)
    ds = ds.map(lambda b: tokenizer(b["answer_text"], truncation=True, max_length=256), batched=True)
    trainer = Trainer(model=model, args=TrainingArguments(output_dir="tmp-eval", report_to=[]))
    pred = np.argmax(trainer.predict(ds).predictions, axis=1)
    gold = test.labels.to_numpy()
    print({"accuracy": accuracy_score(gold, pred),
           "macro_f1": f1_score(gold, pred, average="macro", zero_division=0),
           "quadratic_kappa": cohen_kappa_score(gold, pred, weights="quadratic")})

if __name__ == "__main__":
    main()
