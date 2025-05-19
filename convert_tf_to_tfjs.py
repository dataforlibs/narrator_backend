import tensorflow as tf
import os
import sys
import argparse

def convert_saved_model_to_tfjs(saved_model_dir, output_dir, quantize=False):
    """
    Convert TensorFlow SavedModel to TensorFlow.js format
    
    Args:
        saved_model_dir: Directory containing the SavedModel
        output_dir: Directory to save the converted TensorFlow.js model
        quantize: Whether to apply quantization to reduce model size
    """
    try:
        import tensorflowjs as tfjs
    except ImportError:
        print("tensorflowjs not found. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflowjs"])
        import tensorflowjs as tfjs
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Conversion options
    conversion_options = {
        'input_format': 'tf_saved_model',
        'output_format': 'tfjs_graph_model',
        'signature_name': 'serving_default',
        'saved_model_tags': 'serve'
    }
    
    # Add quantization if requested
    if quantize:
        conversion_options['quantization_bytes'] = 2
    
    print(f"Converting SavedModel to TensorFlow.js format...")
    tfjs.converters.convert_tf_saved_model(
        saved_model_dir,
        output_dir,
        **conversion_options
    )
    
    print(f"TensorFlow.js model saved to {output_dir}")
    print("\nTo use this model in your web application:")
    print("model = await tf.loadGraphModel('./model_directory/model.json');")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert TensorFlow SavedModel to TensorFlow.js format')
    parser.add_argument('--saved_model_dir', required=True, help='Directory containing the SavedModel')
    parser.add_argument('--output_dir', required=True, help='Directory to save the TensorFlow.js model')
    parser.add_argument('--quantize', action='store_true', help='Apply quantization to reduce model size')
    
    args = parser.parse_args()
    
    convert_saved_model_to_tfjs(args.saved_model_dir, args.output_dir, args.quantize)
