import os
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflowjs as tfjs

# Load model and tokenizer from Hugging Face
model_name = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)

# Save tokenizer files (we'll need these for preprocessing)
output_dir = "emotion-roberta-tfjs"
os.makedirs(output_dir, exist_ok=True)
tokenizer.save_pretrained(output_dir)

# Save model in SavedModel format first (intermediate step)
saved_model_path = os.path.join(output_dir, "saved_model")
model.save_pretrained(saved_model_path, saved_model=True)

# Convert the SavedModel to TensorFlow.js format
tfjs_output_dir = os.path.join(output_dir, "tfjs")
tfjs.converters.convert_tf_saved_model(
    saved_model_path,
    tfjs_output_dir
)

print(f"Model successfully converted to TensorFlow.js format in {tfjs_output_dir}")