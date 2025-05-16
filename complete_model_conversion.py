import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import onnx
import onnxruntime as ort
import numpy as np
import os

print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"ONNX Runtime version: {ort.__version__}")

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

# Create fixed input shapes for more reliable web conversion
sequence_length = 16
batch_size = 1

# Create dummy input
dummy_input_ids = torch.ones(batch_size, sequence_length, dtype=torch.long)
dummy_attention_mask = torch.ones(batch_size, sequence_length, dtype=torch.long)

# Export model to ONNX format
onnx_path = "./web_model/emotion_model_web.onnx"

# Use a very basic configuration for maximum compatibility
torch.onnx.export(
    model,                                        
    (dummy_input_ids, dummy_attention_mask),      
    onnx_path,                                    
    export_params=True,                           
    opset_version=11,  # Lower version for better compatibility
    do_constant_folding=True,                      
    input_names=['input_ids', 'attention_mask'],  
    output_names=['logits'],
    # No dynamic axes - fixed sizes for web for maximum compatibility
)

print(f"Model exported to {onnx_path}")

# Verify the model
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("ONNX model verification passed!")

# Check model with onnxruntime
test_session = ort.InferenceSession(onnx_path)
test_inputs = {
    'input_ids': dummy_input_ids.numpy(),
    'attention_mask': dummy_attention_mask.numpy()
}
test_outputs = test_session.run(None, test_inputs)
print("Test inference successful!")
print(f"Output shape: {test_outputs[0].shape}")

# Print label mapping for reference
id2label = model.config.id2label
print("\nEmotion Labels:")
for id, label in id2label.items():
    print(f"  {id}: {label}")

# Create a simple example function to test with Python
def predict_emotion(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", max_length=sequence_length, truncation=True)
    input_ids = inputs["input_ids"].numpy()
    attention_mask = inputs["attention_mask"].numpy()
    
    # Run inference with ONNX Runtime
    ort_inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
    ort_outputs = test_session.run(None, ort_inputs)
    logits = ort_outputs[0]
    
    # Get prediction
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    predicted_class = np.argmax(probabilities, axis=1)[0]
    
    # Return results
    return {
        "text": text,
        "predicted_emotion": id2label[predicted_class],
        "probabilities": {id2label[i]: float(probabilities[0][i]) for i in range(len(id2label))}
    }

# Test the model with a sample text
test_text = "I'm so happy to see you again after all these years!"
result = predict_emotion(test_text)
print("\nTest prediction:")
print(f"Text: \"{result['text']}\"")
print(f"Predicted emotion: {result['predicted_emotion']}")
print("Probabilities:")
for emotion, prob in result["probabilities"].items():
    print(f"  {emotion}: {prob:.4f}")

print("\nExport complete! The model files are ready for web use in the ./web_model directory.")
