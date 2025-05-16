import os

# Suppress OpenMP library initialization error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime
import numpy as np
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification  # or a more specific class
import sys

app = FastAPI()
model_id = "obuchel1/quantized_emotion_model1"  # ONNX model ID on Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Load the ONNX model using optimum
try:
    if sys.platform == "darwin":
        # Use CPU provider on macOS to avoid "illegal hardware instruction"
        ort_model = ORTModelForSequenceClassification.from_pretrained(model_id, provider="CPUExecutionProvider")
    else:
        ort_model = ORTModelForSequenceClassification.from_pretrained(model_id) #, provider="CUDAExecutionProvider")  # You can try to use CUDA if available
    onnx_session = ort_model.session
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to load ONNX model: {e}")


class PredictionRequest(BaseModel):
    text: str


def preprocess(text: str):
    inputs = tokenizer(text, return_tensors="np")
    return {k: v.astype(np.int64) for k, v in inputs.items()}


def postprocess(outputs):
    # Adjust this based on your model's output
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
        ort_inputs = {
            onnx_session.get_inputs()[0].name: processed_input['input_ids'],
            onnx_session.get_inputs()[1].name: processed_input['attention_mask'],
        }  # Adjust input names
        ort_outputs = onnx_session.run(None, ort_inputs)
        prediction = postprocess(ort_outputs)
        return {'prediction': prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
