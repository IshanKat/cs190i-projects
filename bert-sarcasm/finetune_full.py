from config import *
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer, EvalPrediction
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import default_data_collator
from progress_callback import TQDMProgressBar

print("Loading dataset...")
dataset = load_dataset("tweet_eval", "irony")

# === Tokenizer and preprocessing ===
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

tokenized = dataset.map(preprocess_function, batched=True)
tokenized = tokenized.remove_columns(["text"])
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch")

# Load model (default: all layers trainable)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

# Metrics
def compute_metrics(p: EvalPrediction):
    pred = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, pred), "f1": f1_score(p.label_ids, pred)}

# Training arguments
args = TrainingArguments(
    output_dir=OUTPUT_DIR_FULL,
    eval_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    logging_dir="./logs",
    logging_steps=LOGGING_STEPS,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
    callbacks=[TQDMProgressBar()]
)

trainer.train()
