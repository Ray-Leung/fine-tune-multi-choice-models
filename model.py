
# coding: utf-8

# In[1]:


from datasets import load_dataset
from transformers import AutoTokenizer
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch
import evaluate
import numpy as np
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer


def preprocess(examples):
    # adjust the format of the query & see performance
    queries = [
        [f"Here is a description of the cartoon. {examples['image_description'][i]} {examples['image_uncanny_description'][i]} Here is a caption that corresponds to the cartoon. {choice}" for choice in examples['caption_choices'][i]] for i in range(len(examples['caption_choices']))
    ]
    queries = sum(queries, [])
    # print(queries)
    tokenized_examples = tokenizer(queries, truncation=True)
    return {k: [v[i : i + 5] for i in range(0, len(v), 5)] for k, v in tokenized_examples.items()}

# preprocess(dset['train'][0:1])


# turn A, B, C, D, E to 0, 1, 2, 3, 4
def modify_label(examples):
    return {"label": [ord(label) - ord('A') for label in examples['label']]}


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        # print(batch)
        return batch


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

print('cuda:', torch.cuda.is_available())

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu')


# In[3]:


dset = load_dataset("jmhessel/newyorker_caption_contest", "matching")
# dset = load_dataset("jmhessel/newyorker_caption_contest", "ranking")
# dset = load_dataset("jmhessel/newyorker_caption_contest", "explanation")


#tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


# try other models too
#model = AutoModelForMultipleChoice.from_pretrained("bert-base-uncased").to(device)
model = AutoModelForMultipleChoice.from_pretrained("bert-base-cased").to(device)


tokenized_dset = dset.map(preprocess, batched=True)
accuracy = evaluate.load("accuracy")

tokenized_dset = tokenized_dset.map(modify_label, batched=True)

# adjust these params
training_args = TrainingArguments(
    output_dir="./FinetunedBert_lr1e-5_wd0.01",
    evaluation_strategy="steps",
    eval_steps=612,
    save_strategy="steps",
    save_steps=3060,
    logging_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True,
    learning_rate=0.85e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=100,
    weight_decay=0.01
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dset["train"],
    eval_dataset=tokenized_dset["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)



trainer.train()
trainer.evaluate(eval_dataset=tokenized_dset['test'])
# The accuracy reported in the essay is the average of 5-fold cross-validation accuracy rather than the test accuracy.
