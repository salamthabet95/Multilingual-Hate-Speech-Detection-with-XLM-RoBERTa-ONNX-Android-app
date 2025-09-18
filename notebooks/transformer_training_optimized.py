"""
Optimized multilingual transformer training with better hyperparameters
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class MultilingualHateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_balanced_dataset(X_train, y_train, target_ratio=0.45):
    """Create balanced dataset with better ratio"""
    unique_classes, counts = np.unique(y_train, return_counts=True)
    print(f"Original class distribution: {dict(zip(unique_classes, counts))}")
    
    minority_class = unique_classes[np.argmin(counts)]
    majority_class = unique_classes[np.argmax(counts)]
    
    minority_indices = np.where(y_train == minority_class)[0]
    majority_indices = np.where(y_train == majority_class)[0]
    
    minority_count = len(minority_indices)
    majority_count = len(majority_indices)
    
    # Better balance (45% hate, 55% non-hate)
    target_minority_count = int(majority_count * target_ratio / (1 - target_ratio))
    
    if target_minority_count > minority_count:
        oversample_count = target_minority_count - minority_count
        oversample_indices = np.random.choice(minority_indices, size=oversample_count, replace=True)
        balanced_indices = np.concatenate([majority_indices, minority_indices, oversample_indices])
    else:
        target_majority_count = int(minority_count * (1 - target_ratio) / target_ratio)
        majority_indices = np.random.choice(majority_indices, size=min(target_majority_count, majority_count), replace=False)
        balanced_indices = np.concatenate([majority_indices, minority_indices])
    
    np.random.shuffle(balanced_indices)
    
    X_balanced = [X_train[i] for i in balanced_indices]
    y_balanced = [y_train[i] for i in balanced_indices]
    
    print(f"Balanced class distribution: {np.bincount(y_balanced)}")
    return X_balanced, y_balanced

def train_multilingual_transformer_optimized(X_train, X_val, X_test, y_train, y_val, y_test, 
                                           model_name="xlm-roberta-base", max_length=128, 
                                           batch_size=16, learning_rate=1e-5, num_epochs=5):
    """
    Optimized multilingual transformer training
    """
    print(f"Using model: {model_name}")
    print("Training for Arabic, Turkish, and English languages (OPTIMIZED)...")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        problem_type="single_label_classification"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Model loaded on device: {device}")
    
    # Create balanced dataset with better ratio
    print("Creating balanced multilingual training dataset...")
    X_train_balanced, y_train_balanced = create_balanced_dataset(X_train, y_train, target_ratio=0.45)
    
    # Create datasets
    train_dataset = MultilingualHateSpeechDataset(X_train_balanced, y_train_balanced, tokenizer, max_length)
    val_dataset = MultilingualHateSpeechDataset(X_val, y_val, tokenizer, max_length)
    test_dataset = MultilingualHateSpeechDataset(X_test, y_test, tokenizer, max_length)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Compute metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='macro')
        recall = recall_score(labels, predictions, average='macro')
        f1 = f1_score(labels, predictions, average='macro')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    # Optimized training arguments
    training_args = TrainingArguments(
        output_dir='./transformer_output',
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=200,  # More warmup
        weight_decay=0.01,
        learning_rate=learning_rate,  # Lower learning rate
        logging_dir='./logs',
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,  # Keep more checkpoints
        report_to=None,
        remove_unused_columns=False,
        dataloader_drop_last=False,
        gradient_accumulation_steps=1,
        fp16=True,
        dataloader_num_workers=2,
        seed=42,
        # Early stopping
        eval_steps=500,
        save_steps=500
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    # Train the model
    print("Starting optimized multilingual training...")
    print(f"Class distribution in balanced training set: {np.bincount(y_train_balanced)}")
    
    try:
        trainer.train()
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed: {e}")
        print("Continuing with current model state...")
    
    # Test the model
    print("\nTesting model with multilingual examples...")
    model.eval()
    with torch.no_grad():
        test_texts = [
            "This is a normal message",
            "هذا رسالة عادية",  # Arabic
            "Bu normal bir mesaj",  # Turkish
            "I hate you so much",
            "You are stupid and worthless",  # More explicit hate
            "I love this beautiful day"  # Positive sentiment
        ]
        
        for text in test_texts:
            encoding = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
            encoding = {k: v.to(device) for k, v in encoding.items()}
            output = model(**encoding)
            pred = torch.argmax(output.logits, dim=1).item()
            probs = torch.softmax(output.logits, dim=1)
            print(f"Text: {text[:40]}... -> {pred} (0=non-hate, 1=hate), probs: {probs.cpu().numpy()}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"\nTest Set Results:")
    for key, value in test_results.items():
        print(f"{key}: {value:.4f}")
    
    # Get predictions for confusion matrix
    test_predictions = trainer.predict(test_dataset)
    y_test_pred_transformer = np.argmax(test_predictions.predictions, axis=1)
    
    # Calculate metrics
    test_accuracy_transformer = accuracy_score(y_test, y_test_pred_transformer)
    test_precision_macro_transformer = precision_score(y_test, y_test_pred_transformer, average='macro')
    test_precision_weighted_transformer = precision_score(y_test, y_test_pred_transformer, average='weighted')
    test_recall_macro_transformer = recall_score(y_test, y_test_pred_transformer, average='macro')
    test_recall_weighted_transformer = recall_score(y_test, y_test_pred_transformer, average='weighted')
    test_f1_macro_transformer = f1_score(y_test, y_test_pred_transformer, average='macro')
    test_f1_weighted_transformer = f1_score(y_test, y_test_pred_transformer, average='weighted')
    
    print(f"\nDetailed Test Results:")
    print(f"Accuracy: {test_accuracy_transformer:.4f}")
    print(f"Precision (macro): {test_precision_macro_transformer:.4f}")
    print(f"Precision (weighted): {test_precision_weighted_transformer:.4f}")
    print(f"Recall (macro): {test_recall_macro_transformer:.4f}")
    print(f"Recall (weighted): {test_recall_weighted_transformer:.4f}")
    print(f"F1 (macro): {test_f1_macro_transformer:.4f}")
    print(f"F1 (weighted): {test_f1_weighted_transformer:.4f}")
    
    # Confusion matrix
    cm_transformer = confusion_matrix(y_test, y_test_pred_transformer)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_transformer, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Hate', 'Hate'], 
                yticklabels=['Non-Hate', 'Hate'])
    plt.title('Confusion Matrix - XLM-RoBERTa (Optimized)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/confusion_transformer_optimized.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results
    transformer_test_results = {
        'model': 'XLM-RoBERTa-Multilingual-Optimized',
        'accuracy': test_accuracy_transformer,
        'precision_macro': test_precision_macro_transformer,
        'precision_weighted': test_precision_weighted_transformer,
        'recall_macro': test_recall_macro_transformer,
        'recall_weighted': test_recall_weighted_transformer,
        'f1_macro': test_f1_macro_transformer,
        'f1_weighted': test_f1_weighted_transformer
    }
    
    return model, tokenizer, transformer_test_results, cm_transformer
