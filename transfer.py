import tensorflowjs as tfjs
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('tf_model.h5')

# Convert and save
tfjs.converters.save_keras_model(model, 'output_folder')
