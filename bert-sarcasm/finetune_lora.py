from config import *
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer, EvalPrediction
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import default_data_collator
from progress_callback import TQDMProgressBar

dataset = load_dataset("tweet_eval", "irony")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
tokenized = dataset.map(lambda x: tokenizer(x["text"], truncation=True), batched=True)

model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=LORA_RANK, lora_alpha=LORA_ALPHA,
                         lora_dropout=LORA_DROPOUT, bias=LORA_BIAS)
model = get_peft_model(model, lora_config)

def compute_metrics(p: EvalPrediction):
    pred = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, pred), "f1": f1_score(p.label_ids, pred)}

args = TrainingArguments(output_dir=OUTPUT_DIR_LORA, eval_strategy="epoch", learning_rate=LEARNING_RATE,
                         per_device_train_batch_size=TRAIN_BATCH_SIZE, per_device_eval_batch_size=EVAL_BATCH_SIZE,
                         num_train_epochs=NUM_EPOCHS, weight_decay=WEIGHT_DECAY,
                         logging_dir="./logs", logging_steps=LOGGING_STEPS)

trainer = Trainer(model=model, args=args, train_dataset=tokenized["train"], eval_dataset=tokenized["validation"],
                  tokenizer=tokenizer, data_collator=default_data_collator, compute_metrics=compute_metrics, callbacks=[TQDMProgressBar()])

trainer.train()
