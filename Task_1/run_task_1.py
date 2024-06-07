import torch
from transformers import AutoModel, AutoTokenizer

class SentenceTransformer:
    def __init__(self):
        model_name='bert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode(self, sentences):
        # Tokenize input sentences and create attention masks
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask'] #Attention mask in order to ignore the padded areas during embedding processing

        # Get model outputs using input ids and attention masks
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Using [CLS] token's embedding as sentence embedding, in order to get fixed length embedding
        embeddings = outputs.last_hidden_state[:, 0, :] # Extract the [CLS] token's embeddings at Index 0
        return embeddings

# Initialize the Sentence Transformer
sentence_transformer = SentenceTransformer()

# Example sentences
sentences = ["This is a sample sentence.", "Here's another example.", "Deep learning transforms everything."]

# Encode sentences
embeddings = sentence_transformer.encode(sentences)
print("Generated Embeddings:", embeddings)
