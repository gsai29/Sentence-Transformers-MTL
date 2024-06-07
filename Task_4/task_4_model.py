
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import torch.optim as optim

class MultiTaskSentenceTransformer(nn.Module):
    def __init__(self):
        model_name='bert-base-uncased',
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        # Task A: Sentence Classification could be something like Question, Statement, Command
        num_classes = 3  # Example labels: 0 = 'Question', 1 = 'Statement', 2 = 'Command'
        # Task B: Sentiment Analysis could be Positive, Negative, Neutral
        num_sentiments = 3  # Example labels: 0 = 'Positive', 1 = 'Negative', 2 = 'Neutral'
        
        # Task-specific heads:
        self.classification_head = nn.Linear(768, num_classes)
        self.sentiment_head = nn.Linear(768, num_sentiments)

    def forward(self, sentences, task="classification"):
        # Tokenize input sentences and generate attention masks
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask']

        # Get embeddings from BERT using [CLS] token's output
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Apply the appropriate task-specific head
        if task == "classification":
            logits = self.classification_head(cls_embedding)
            return F.softmax(logits, dim=1)
        elif task == "sentiment":
            logits = self.sentiment_head(cls_embedding)
            return F.softmax(logits, dim=1)

# Example usage
model = MultiTaskSentenceTransformer()
sentences = ["This is a great movie!", "What time does the store close?"]
classification_probs = model(sentences, task="classification")
print("Classification Probabilities:", classification_probs)


# Optimizer setup with layer-wise learning rates
optimizer = optim.Adam([
    {'params': model.bert.parameters(), 'lr': 1e-5},  # Lower learning rate for BERT parameters
    {'params': model.classification_head.parameters(), 'lr': 3e-4},  # Higher learning rate for the classification head
    {'params': model.sentiment_head.parameters(), 'lr': 3e-4}  # Higher learning rate for the sentiment analysis head
])

# Training loop would go here, including loss calculation and calling optimizer.step()
