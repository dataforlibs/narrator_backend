# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import onnxruntime as ort
from transformers import AutoTokenizer
import os
import torch.nn.functional as F
from typing import Dict, List

# Define the request model
class TextRequest(BaseModel):
    text: str

# Define the response model
class EmotionResponse(BaseModel):
    predicted_emotion: str
    probabilities: Dict[str, float]

# Initialize FastAPI app
app = FastAPI(title="Emotion Analysis API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the tokenizer and ONNX model
model_dir = os.path.dirname(os.path.abspath(__file__))
tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "emotion_tokenizer"))
session = ort.InferenceSession(os.path.join(model_dir, "emotion_model.onnx"))

# Define emotion labels
emotion_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

@app.post("/predict", response_model=EmotionResponse)
async def predict(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Tokenize the input text
    inputs = tokenizer(
        request.text, 
        return_tensors="pt", 
        truncation=True, 
        padding="max_length", 
        max_length=16
    )
    
    # Run inference
    ort_inputs = {
        'input_ids': inputs['input_ids'].numpy(),
        'attention_mask': inputs['attention_mask'].numpy()
    }
    
    ort_outputs = session.run(['logits'], ort_inputs)
    logits = torch.tensor(ort_outputs[0])
    
    # Get probabilities and predicted class
    probabilities = F.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    
    # Return predicted emotion and probabilities
    emotion_probs = {emotion_labels[i]: float(probabilities[0][i]) for i in range(len(emotion_labels))}
    
    return EmotionResponse(
        predicted_emotion=emotion_labels[predicted_class],
        probabilities=emotion_probs
    )

@app.get("/")
async def root():
    return {"message": "Welcome to the Emotion Analysis API. Use /predict endpoint to analyze text."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
