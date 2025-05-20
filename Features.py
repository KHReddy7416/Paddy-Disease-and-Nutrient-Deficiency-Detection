import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import requests
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from io import BytesIO
from PIL import Image

# Step 1: Download the Image
img_url = "Nitro_1.jpg"

#response = requests.get(img_url)
#img = Image.open(BytesIO(response.content))
img = Image.open(img_url)
img = img.resize((224, 224))
img.save("test_image.jpg")  # Save image locally

# Step 2: Load MobileNetV3-Large Model
base_model = MobileNetV3Large(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Step 3: Select Convolutional Layers for Feature Extraction
conv_layers = [(layer.name, layer.output) for layer in base_model.layers if 'conv' in layer.name]
layer_names = [name for name, _ in conv_layers]  # Extract layer names

# Create a feature extractor model
feature_extractor = Model(inputs=base_model.input, outputs=[output for _, output in conv_layers])

# Step 4: Load and Preprocess Image
img = image.load_img("test_image.jpg", target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # Normalize pixel values

# Step 5: Get Feature Maps
feature_maps = feature_extractor.predict(img_array)

# Step 6: Define Layer Functionality Descriptions
layer_info = {
    0: "Basic Edge Detection",
    5: "Texture & Pattern Recognition",
    10: "Shape Detection",
    15: "Curves & High-Level Features",
    25: "Final Deep Features for Classification"
}

# Step 7: Visualize Feature Maps from Selected Layers
selected_layers = [0, 5, 10, 15, 25]  # Selected key layers for visualization

for index in selected_layers:
    if index >= len(feature_maps):
        continue  # Avoid accessing non-existing layers

    feature_map = feature_maps[index]
    num_features = feature_map.shape[-1]  # Number of feature channels
    layer_name = layer_names[index]  # Get layer name

    plt.figure(figsize=(15, 10))
    for i in range(min(num_features, 16)):  # Show up to 16 feature maps per layer
        plt.subplot(4, 4, i + 1)
        plt.imshow(feature_map[0, :, :, i], cmap='viridis')
        plt.axis('off')
    
    # Display the layer name and functionality
    plt.suptitle(f"Layer {index + 1}: {layer_name}\nFunction: {layer_info.get(index, 'Advanced Feature Extraction')}", fontsize=14)
    plt.show()
