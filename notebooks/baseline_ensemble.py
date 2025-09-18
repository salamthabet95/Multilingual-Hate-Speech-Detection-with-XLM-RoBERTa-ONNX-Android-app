"""
Ensemble baseline model training combining multiple algorithms for better performance
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def train_ensemble_baseline_models(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Train ensemble baseline models combining multiple algorithms
    """
    print("Training Ensemble Baseline Models...")
    
    # TF-IDF Vectorizer with optimized parameters
    vectorizer = TfidfVectorizer(
        max_features=15000,  # More features for ensemble
        ngram_range=(1, 3),  # Include trigrams
        stop_words='english',
        lowercase=True,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True  # Use sublinear TF scaling
    )
    
    # Individual models with optimized parameters
    models = {
        'LogisticRegression': Pipeline([
            ('tfidf', vectorizer),
            ('classifier', LogisticRegression(
                random_state=42, 
                max_iter=2000,
                class_weight='balanced',
                C=1.0
            ))
        ]),
        'SVC': Pipeline([
            ('tfidf', vectorizer),
            ('classifier', SVC(
                random_state=42, 
                class_weight='balanced',
                C=1.0,
                probability=True,  # Enable probability estimates
                kernel='linear'  # Use linear kernel for speed
            ))
        ]),
        'RandomForest': Pipeline([
            ('tfidf', vectorizer),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ))
        ])
    }
    
    # Parameter grids for individual models
    param_grids = {
        'LogisticRegression': {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__solver': ['liblinear', 'lbfgs']
        },
        'SVC': {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__kernel': ['linear', 'rbf']
        },
        'RandomForest': {
            'classifier__n_estimators': [50, 100, 150],
            'classifier__max_depth': [10, 15, 20],
            'classifier__min_samples_split': [2, 5, 10]
        }
    }
    
    # Train individual models
    individual_models = {}
    individual_results = {}
    
    print("\n=== Training Individual Models ===")
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        grid_search = GridSearchCV(
            model, 
            param_grids[name], 
            cv=3,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit on training data
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Evaluate on validation set
        y_val_pred = grid_search.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_precision = precision_score(y_val, y_val_pred, average='macro')
        val_recall = recall_score(y_val, y_val_pred, average='macro')
        val_f1 = f1_score(y_val, y_val_pred, average='macro')
        
        print(f"Validation - Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
        
        # Evaluate on test set
        y_test_pred = grid_search.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision_macro = precision_score(y_test, y_test_pred, average='macro')
        test_precision_weighted = precision_score(y_test, y_test_pred, average='weighted')
        test_recall_macro = recall_score(y_test, y_test_pred, average='macro')
        test_recall_weighted = recall_score(y_test, y_test_pred, average='weighted')
        test_f1_macro = f1_score(y_test, y_test_pred, average='macro')
        test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted')
        
        print(f"Test - Accuracy: {test_accuracy:.4f}, F1 (macro): {test_f1_macro:.4f}, F1 (weighted): {test_f1_weighted:.4f}")
        
        # Store results
        individual_models[name] = grid_search.best_estimator_
        individual_results[name] = {
            'accuracy': test_accuracy,
            'precision_macro': test_precision_macro,
            'precision_weighted': test_precision_weighted,
            'recall_macro': test_recall_macro,
            'recall_weighted': test_recall_weighted,
            'f1_macro': test_f1_macro,
            'f1_weighted': test_f1_weighted
        }
    
    # Create ensemble model
    print("\n=== Creating Ensemble Model ===")
    
    # Voting Classifier with soft voting (uses predicted probabilities)
    ensemble_model = VotingClassifier(
        estimators=[
            ('lr', individual_models['LogisticRegression']),
            ('svc', individual_models['SVC']),
            ('rf', individual_models['RandomForest'])
        ],
        voting='soft'  # Use predicted probabilities for better performance
    )
    
    # Train ensemble
    print("Training ensemble model...")
    ensemble_model.fit(X_train, y_train)
    
    # Evaluate ensemble on validation set
    y_val_pred_ensemble = ensemble_model.predict(X_val)
    val_accuracy_ensemble = accuracy_score(y_val, y_val_pred_ensemble)
    val_precision_ensemble = precision_score(y_val, y_val_pred_ensemble, average='macro')
    val_recall_ensemble = recall_score(y_val, y_val_pred_ensemble, average='macro')
    val_f1_ensemble = f1_score(y_val, y_val_pred_ensemble, average='macro')
    
    print(f"Ensemble Validation - Accuracy: {val_accuracy_ensemble:.4f}, Precision: {val_precision_ensemble:.4f}, Recall: {val_recall_ensemble:.4f}, F1: {val_f1_ensemble:.4f}")
    
    # Evaluate ensemble on test set
    y_test_pred_ensemble = ensemble_model.predict(X_test)
    test_accuracy_ensemble = accuracy_score(y_test, y_test_pred_ensemble)
    test_precision_macro_ensemble = precision_score(y_test, y_test_pred_ensemble, average='macro')
    test_precision_weighted_ensemble = precision_score(y_test, y_test_pred_ensemble, average='weighted')
    test_recall_macro_ensemble = recall_score(y_test, y_test_pred_ensemble, average='macro')
    test_recall_weighted_ensemble = recall_score(y_test, y_test_pred_ensemble, average='weighted')
    test_f1_macro_ensemble = f1_score(y_test, y_test_pred_ensemble, average='macro')
    test_f1_weighted_ensemble = f1_score(y_test, y_test_pred_ensemble, average='weighted')
    
    print(f"Ensemble Test - Accuracy: {test_accuracy_ensemble:.4f}, F1 (macro): {test_f1_macro_ensemble:.4f}, F1 (weighted): {test_f1_weighted_ensemble:.4f}")
    
    # Compare individual vs ensemble performance
    print("\n=== Performance Comparison ===")
    print("Model\t\t\tF1 (macro)\tF1 (weighted)")
    print("-" * 50)
    for name, results in individual_results.items():
        print(f"{name:<20}\t{results['f1_macro']:.4f}\t\t{results['f1_weighted']:.4f}")
    
    ensemble_results = {
        'accuracy': test_accuracy_ensemble,
        'precision_macro': test_precision_macro_ensemble,
        'precision_weighted': test_precision_weighted_ensemble,
        'recall_macro': test_recall_macro_ensemble,
        'recall_weighted': test_recall_weighted_ensemble,
        'f1_macro': test_f1_macro_ensemble,
        'f1_weighted': test_f1_weighted_ensemble
    }
    
    print(f"{'Ensemble':<20}\t{ensemble_results['f1_macro']:.4f}\t\t{ensemble_results['f1_weighted']:.4f}")
    
    # Confusion matrix for ensemble
    cm_ensemble = confusion_matrix(y_test, y_test_pred_ensemble)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_ensemble, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Hate', 'Hate'], 
                yticklabels=['Non-Hate', 'Hate'])
    plt.title('Confusion Matrix - Ensemble Baseline Models')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    # Plot performance comparison
    models_names = list(individual_results.keys()) + ['Ensemble']
    f1_scores = [individual_results[name]['f1_macro'] for name in individual_results.keys()] + [ensemble_results['f1_macro']]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models_names, f1_scores, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    plt.title('F1 Score Comparison: Individual vs Ensemble Models')
    plt.ylabel('F1 Score (Macro)')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.4f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Determine best model
    best_individual_f1 = max(individual_results.values(), key=lambda x: x['f1_macro'])['f1_macro']
    best_individual_name = max(individual_results.keys(), key=lambda x: individual_results[x]['f1_macro'])
    
    if ensemble_results['f1_macro'] > best_individual_f1:
        best_model = ensemble_model
        best_results = ensemble_results
        best_cm = cm_ensemble
        print(f"\n🏆 Best Model: Ensemble (F1: {ensemble_results['f1_macro']:.4f})")
        print(f"   Improvement over best individual ({best_individual_name}): {ensemble_results['f1_macro'] - best_individual_f1:.4f}")
    else:
        best_model = individual_models[best_individual_name]
        best_results = individual_results[best_individual_name]
        best_cm = confusion_matrix(y_test, individual_models[best_individual_name].predict(X_test))
        print(f"\n🏆 Best Model: {best_individual_name} (F1: {best_individual_f1:.4f})")
    
    return best_model, best_results, best_cm, individual_results, ensemble_results
