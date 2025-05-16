import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import json

print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")

# Create output directory
os.makedirs('web_model', exist_ok=True)

# Download model and tokenizer
model_id = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
model.eval()

# Save tokenizer files (including vocab)
tokenizer.save_pretrained("./web_model/tokenizer")
print("Tokenizer files saved to ./web_model/tokenizer")

# Save the model directly in PyTorch format
torch_path = "./web_model/emotion_model.pt"
torch.save(model.state_dict(), torch_path)
print(f"PyTorch model saved to {torch_path}")

# Trace the model with TorchScript (better for JavaScript conversion)
sequence_length = 16
batch_size = 1

# Create dummy input
dummy_input_ids = torch.ones(batch_size, sequence_length, dtype=torch.long)
dummy_attention_mask = torch.ones(batch_size, sequence_length, dtype=torch.long)

# Create a traced model
traced_model = torch.jit.trace(model, (dummy_input_ids, dummy_attention_mask))
traced_path = "./web_model/emotion_model_traced.pt"
traced_model.save(traced_path)
print(f"TorchScript traced model saved to {traced_path}")

# Save the model configuration
config_path = "./web_model/model_config.json"
with open(config_path, 'w') as f:
    # Extract relevant configuration
    config_dict = {
        "id2label": model.config.id2label,
        "sequence_length": sequence_length,
        "model_name": model_id,
        "num_labels": len(model.config.id2label)
    }
    json.dump(config_dict, f, indent=2)
print(f"Model configuration saved to {config_path}")

# Create a simple example function to test with Python
def predict_emotion_torch(text):
    # Tokenize input text
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        padding="max_length", 
        max_length=sequence_length, 
        truncation=True
    )
    
    # Move inputs to the same device as model
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Run inference with PyTorch
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits.numpy()
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()
    predicted_class = probabilities.argmax(axis=1)[0]
    
    # Return results
    id2label = model.config.id2label
    return {
        "text": text,
        "predicted_emotion": id2label[predicted_class],
        "probabilities": {id2label[i]: float(probabilities[0][i]) for i in range(len(id2label))}
    }

# Test the model with a sample text
test_text = "I'm so happy to see you again after all these years!"
result = predict_emotion_torch(test_text)
print("\nTest prediction:")
print(f"Text: \"{result['text']}\"")
print(f"Predicted emotion: {result['predicted_emotion']}")
print("Probabilities:")
for emotion, prob in result["probabilities"].items():
    print(f"  {emotion}: {prob:.4f}")

# Save JavaScript-friendly model format (if tensorflowjs is available)
try:
    import tensorflowjs as tfjs
    from transformers import TFAutoModelForSequenceClassification
    
    # Convert to TensorFlow format first
    tf_model_path = "./web_model/tf_model"
    os.makedirs(tf_model_path, exist_ok=True)
    
    # Create a TF version of the model
    tf_model = TFAutoModelForSequenceClassification.from_pretrained(
        model_id, 
        from_pt=True
    )
    
    # Save as TF SavedModel format
    tf_model.save_pretrained(tf_model_path)
    
    # Convert to TensorFlow.js format
    tfjs_path = "./web_model/tfjs_model"
    tfjs.converters.save_keras_model(tf_model, tfjs_path)
    print(f"\nTensorFlow.js model saved to {tfjs_path}")
    
except ImportError:
    print("\nTensorFlow/TensorFlow.js not installed. Skipping TF.js conversion.")
    print("To convert to TF.js format, install with: pip install tensorflowjs tensorflow")

print("\nExport complete! The model files are ready for web use in the ./web_model directory.")
print("\nFor web deployment, you can use either:")
print("1. The TorchScript model with ONNX Web Runtime")
print("2. The TensorFlow.js model (if conversion was successful)")
print("3. Convert the PyTorch model with a tool like Netron (https://netron.app)")
