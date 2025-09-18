"""
Model export utilities for PyTorch, ONNX, and TorchScript
"""

import os
import pickle
import torch
import onnx
from onnxruntime import InferenceSession
import numpy as np

def save_baseline_model(model, model_path="models/baseline/baseline_model.pkl"):
    """Save baseline model as pickle file"""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Baseline model saved: {model_path}")

def save_huggingface_model(model, tokenizer, model_path="models/transformer/hf_model"):
    """Save full HuggingFace model and tokenizer"""
    os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"HuggingFace model saved: {model_path}/")
    print("Files saved:")
    for file in os.listdir(model_path):
        print(f"  - {file}")

class TorchScriptWrapper(torch.nn.Module):
    """Wrapper class to make HuggingFace model TorchScript compatible"""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, input_ids, attention_mask):
        # Call the model and return only logits (not the full dict)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

def save_torchscript_model(model, tokenizer, model_path="models/transformer/torchscript/model.pt", max_length=128):
    """Export model to TorchScript format"""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Move model to CPU for export
    device = next(model.parameters()).device
    model_cpu = model.cpu()
    
    # Create wrapper for TorchScript compatibility
    wrapped_model = TorchScriptWrapper(model_cpu)
    wrapped_model.eval()
    
    # Create example input for tracing (on CPU)
    example_text = "This is a sample text for tracing"
    example_encoding = tokenizer(
        example_text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Ensure inputs are on CPU
    input_ids = example_encoding['input_ids'].cpu()
    attention_mask = example_encoding['attention_mask'].cpu()
    
    # Trace the wrapped model
    with torch.no_grad():
        traced_model = torch.jit.trace(
            wrapped_model,
            (input_ids, attention_mask)
        )
    
    # Save TorchScript model
    torch.jit.save(traced_model, model_path)
    print(f"TorchScript model saved: {model_path}")
    
    # Move model back to original device
    model.to(device)

def save_onnx_model(model, model_path="models/transformer/onnx/model.onnx", max_length=128):
    """Export model to ONNX format"""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Move model to CPU for export
    device = next(model.parameters()).device
    model_cpu = model.cpu()
    
    # Create wrapper for ONNX compatibility
    wrapped_model = TorchScriptWrapper(model_cpu)
    wrapped_model.eval()
    
    # Create dummy inputs for ONNX export (on CPU)
    dummy_input_ids = torch.randint(0, 1000, (1, max_length), dtype=torch.long)
    dummy_attention_mask = torch.ones(1, max_length, dtype=torch.long)
    
    # Export to ONNX
    with torch.no_grad():
        torch.onnx.export(
            wrapped_model,
            (dummy_input_ids, dummy_attention_mask),
            model_path,
            export_params=True,
            opset_version=14,  # Updated to support scaled_dot_product_attention
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size'}
            }
        )
    print(f"ONNX model saved: {model_path}")
    
    # Move model back to original device
    model.to(device)
    
    # Verify ONNX model
    try:
        onnx_model = onnx.load(model_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification: PASSED")
    except Exception as e:
        print(f"ONNX model verification: FAILED - {e}")

def test_onnx_model(model_path, tokenizer, test_text="This is a test", max_length=128):
    """Test ONNX model with sample input"""
    try:
        # Load ONNX model
        session = InferenceSession(model_path)
        
        # Prepare input
        encoding = tokenizer(
            test_text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='np'
        )
        
        # Run inference
        inputs = {
            'input_ids': encoding['input_ids'].astype(np.int64),
            'attention_mask': encoding['attention_mask'].astype(np.int64)
        }
        
        outputs = session.run(None, inputs)
        logits = outputs[0]
        prediction = np.argmax(logits, axis=1)[0]
        confidence = float(np.max(torch.softmax(torch.tensor(logits), dim=1)))
        
        print(f"ONNX Test - Text: '{test_text}'")
        print(f"Prediction: {prediction} (0=non-hate, 1=hate)")
        print(f"Confidence: {confidence:.4f}")
        
        return prediction, confidence
        
    except Exception as e:
        print(f"ONNX model test failed: {e}")
        return None, None
