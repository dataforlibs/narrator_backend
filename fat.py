import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import onnxruntime
import numpy as np
from transformers import AutoTokenizer

app = FastAPI()
tokenizer = AutoTokenizer.from_pretrained("obuchel1/quantized_emotion_model1")  # Replace
onnx_path = "quantized_emotion_model1/emotion_model.onnx"  # Replace, assumes the onnx model is in a subfolder
ort_session = onnxruntime.InferenceSession(onnx_path)

class PredictionRequest(BaseModel):
    text: str

def preprocess(text: str):
    inputs = tokenizer(text, return_tensors="np")
    return {k: v.astype(np.int64) for k, v in inputs.items()}

def postprocess(outputs):
    # Adjust this based on your model's output structure
    logits = outputs[0]
    predicted_class_id = np.argmax(logits, axis=-1)
    return tokenizer.decode(predicted_class_id)

@app.post('/predict')
async def predict(request: PredictionRequest):
    try:
        text = request.text
        if not text:
            raise HTTPException(status_code=400, detail="No text provided")

        processed_input = preprocess(text)
        ort_inputs = {ort_session.get_inputs()[0].name: processed_input['input_ids'],
                      ort_session.get_inputs()[1].name: processed_input['attention_mask']} # Adjust input names
        ort_outputs = ort_session.run(None, ort_inputs)
        prediction = postprocess(ort_outputs)

        return {'prediction': prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
