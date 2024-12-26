from tensorflow.keras.models import load_model as tf_load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# Load model
def load_model(model_option):
    """Load MobileNet or CNN model based on user selection."""
    if model_option == "MobileNet":
        model_path = r"C:\Users\Zulvikar Harist\Documents\KULIAHHH\ML_UAP\src\models\modelmobilenet.h5"
    elif model_option == "CNN":
        model_path = r"C:\Users\Zulvikar Harist\Documents\KULIAHHH\ML_UAP\src\models\modelcnn.h5"
    else:
        raise ValueError("Model tidak dikenal.")
    return tf_load_model(model_path)

# Preprocess image
def preprocess_image(image, target_size):
    """Resize and normalize the image for model prediction."""
    image = image.resize(target_size)
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0
    return image_array

# Predict health with probabilities
def predict_health_with_probabilities(model, image):
    """Predict health of a bee and return probabilities for all classes."""
    class_labels = [
        "Healthy",
        "Few Varroa, Hive Beetles",
        "Varroa, Small Hive Beetles",
        "Ant Problems",
        "Hive Being Robbed",
        "Missing Queen",
    ]
    predictions = model.predict(image)
    probabilities = {class_labels[i]: predictions[0][i] for i in range(len(class_labels))}
    return probabilities

# Grad-CAM
def generate_grad_cam(model, image, class_index, layer_name):
    """Generate Grad-CAM heatmap for a specific class."""
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

def overlay_heatmap(heatmap, original_image, alpha=0.4):
    """Overlay Grad-CAM heatmap on original image."""
    # Pastikan heatmap memiliki nilai 0-255
    heatmap = np.uint8(255 * heatmap)

    # Ubah heatmap menjadi berwarna
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Ubah gambar asli ke format numpy array
    original_image_np = np.array(original_image)

    # Jika gambar asli dalam mode grayscale, ubah menjadi RGB
    if len(original_image_np.shape) == 2:
        original_image_np = cv2.cvtColor(original_image_np, cv2.COLOR_GRAY2BGR)

    # Sesuaikan ukuran heatmap dengan gambar asli
    heatmap_colored = cv2.resize(heatmap_colored, (original_image_np.shape[1], original_image_np.shape[0]))

    # Gabungkan heatmap dan gambar asli
    overlayed_image = cv2.addWeighted(original_image_np, 1 - alpha, heatmap_colored, alpha, 0)

    return Image.fromarray(overlayed_image)
