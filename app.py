# app.py
from flask import Flask, request, jsonify
import torch
import onnx
import onnxruntime as ort
from transformers import AutoTokenizer
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load the tokenizer and ONNX model
model_dir = os.path.dirname(os.path.abspath(__file__))
tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "emotion_tokenizer"))
session = ort.InferenceSession(os.path.join(model_dir, "emotion_model.onnx"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                      padding="max_length", max_length=16)
    
    # Run inference
    ort_inputs = {
        'input_ids': inputs['input_ids'].numpy(),
        'attention_mask': inputs['attention_mask'].numpy()
    }
    
    ort_outputs = session.run(['logits'], ort_inputs)
    logits = torch.tensor(ort_outputs[0])
    
    # Get probabilities and predicted class
    import torch.nn.functional as F
    probabilities = F.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    
    # Get the emotion labels
    emotion_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    
    # Return predicted emotion and probabilities
    emotion_probs = {emotion_labels[i]: float(probabilities[0][i]) for i in range(len(emotion_labels))}
    result = {
        "predicted_emotion": emotion_labels[predicted_class],
        "probabilities": emotion_probs
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
