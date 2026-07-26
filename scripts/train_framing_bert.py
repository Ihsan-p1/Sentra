"""
Script to fine-tune DistilBERT for Framing Classification (Media Source Prediction).
Target: Satisfy "Fine-tuned Model" requirement for MVP.
"""
import json
import random
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import settings

# --- Config ---
MODEL_SAVE_DIR = os.path.join(settings.MODELS_DIR, "framing_bert")
EPOCHS = 3
BATCH_SIZE = 8
MAX_LEN = 128
LEARNING_RATE = 2e-5
SEED = 42

# Reproducible runs (the evaluation report compares against a fixed baseline)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class ArticleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        
        # Modern tokenizer call API (encode_plus was removed in transformers v5)
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train():
    print("Starting DistilBERT Fine-tuning for Framing Analysis...")
    
    # 1. Load Data from the corpus
    import sys
    sys.path.insert(0, ".")
    from data.corpus import corpus_metadata, load_articles

    articles = load_articles()
    print(f"Loaded {len(articles)} articles ({corpus_metadata()['version']}).")
    
    # 2. Prepare Data (Chunking headlines + snippets for training)
    texts = []
    labels = []
    
    # Derive labels from the canonical source list so they can never drift from
    # the media_source values used at ingestion and retrieval time.
    label_map = {media: idx for idx, media in enumerate(settings.SUPPORTED_MEDIA)}
    print(f"Label map: {label_map}")

    # Reverse map for saving
    id_to_label = {v: k for k, v in label_map.items()}

    skipped = {}
    for art in articles:
        media = art.get('media_source')
        if media not in label_map:
            skipped[media] = skipped.get(media, 0) + 1
            continue

        label = label_map[media]
        
        # Use simple text segments: Title + Content chunks
        texts.append(art['title'])
        labels.append(label)
        
        # Add content chunks (first 300 chars)
        content_snippet = art['content'][:300]
        texts.append(content_snippet)
        labels.append(label)
        
    print(f"Training samples: {len(texts)}")
    if skipped:
        # Loud on purpose: a silent skip here previously dropped a whole source
        # from training while the pipeline still queried it at runtime.
        print(f"[WARN] Skipped articles from sources not in SUPPORTED_MEDIA: {skipped}")

    # 3. Train/Test Split (stratified so every source appears in both splits)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    
    # 4. Initialize Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=len(label_map)
    )
    
    # Use GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    # 5. Data Loaders
    train_dataset = ArticleDataset(X_train, y_train, tokenizer, MAX_LEN)
    test_dataset = ArticleDataset(X_test, y_test, tokenizer, MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # 6. Training Loop
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            model.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Average Loss: {avg_loss:.4f}")
        
    # 7. Evaluation
    print("\nEvaluating...")
    model.eval()
    predictions = []
    real_values = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, preds = torch.max(outputs.logits, dim=1)
            predictions.extend(preds.cpu().tolist())
            real_values.extend(labels.cpu().tolist())
            
    print(classification_report(real_values, predictions, target_names=list(label_map.keys()), labels=list(range(len(label_map))), zero_division=0))
    
    # 8. Save Model
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        
    model.save_pretrained(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)
    
    # Save label map
    with open(os.path.join(MODEL_SAVE_DIR, 'label_map.json'), 'w') as f:
        json.dump(id_to_label, f)
        
    print(f"Model saved to {MODEL_SAVE_DIR}")

if __name__ == "__main__":
    train()
