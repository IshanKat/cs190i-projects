from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer, EvalPrediction
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import default_data_collator

# Load data
dataset = load_dataset("tweet_eval", "sarcasm")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenized = dataset.map(lambda x: tokenizer(x["text"], truncation=True), batched=True)

# Load model (all layers trainable by default)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Metrics
def compute_metrics(p: EvalPrediction):
    pred = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, pred), "f1": f1_score(p.label_ids, pred)}

# Trainer
args = TrainingArguments(output_dir="results_full", evaluation_strategy="epoch", learning_rate=2e-5,
                         per_device_train_batch_size=16, per_device_eval_batch_size=16, num_train_epochs=3,
                         weight_decay=0.01, logging_dir="./logs", logging_steps=10)
trainer = Trainer(model=model, args=args, train_dataset=tokenized["train"], eval_dataset=tokenized["validation"],
                  tokenizer=tokenizer, data_collator=default_data_collator, compute_metrics=compute_metrics)

trainer.train()
