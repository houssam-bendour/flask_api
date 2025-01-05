from flask import Flask, request, jsonify
from keras.applications import DenseNet201
from keras.models import Model
from keras.utils import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from PIL import Image
import pickle

# Load the DenseNet201 model and create the feature extractor
base_model = DenseNet201()
feature_extractor = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

# Load the caption generation model and tokenizer
model_path = "models/caption_model.keras"
tokenizer_path = "models/tokenizer.pkl"
caption_model = load_model(model_path)
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

max_length = 34

# Initialize Flask app
app = Flask(__name__)
def extract_features(img):
    """
    Extract features from a given image using the pre-trained DenseNet201 model.
    """
    # Convert the image to a NumPy array
    img = img.resize((224, 224))  # Resize to 224x224
    img = img_to_array(img)
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Use the feature extractor model to predict features
    feature = feature_extractor.predict(img, verbose=0)

    return feature[0]  # Return the feature vector

def idx_to_word(integer,tokenizer):

    for word, index in tokenizer.word_index.items():
        if index==integer:
            return word
    return None

def predict_caption(model, tokenizer, max_length, feature):

    feature = np.expand_dims(feature, axis=0)
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)

        y_pred = model.predict([feature,sequence])
        y_pred = np.argmax(y_pred)

        print("Feature shape:", feature.shape)
        print("Sequence shape:", sequence.shape)

        word = idx_to_word(y_pred, tokenizer)

        if word is None:
            break

        in_text+= " " + word

        if word == 'endseq':
            break

    # Remove "startseq" and "endseq" from the caption
    caption = in_text.split(" ")[1:-1]  # Split into words and remove the first and last tokens
    return " ".join(caption)  # Rejoin the remaining words into a single string

@app.route("/")
def home():
    return "Hello from Flask on Vercel!"

@app.route("/generate_caption", methods=["POST"])
def generate_caption():
    """
    API endpoint to generate captions for uploaded images.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Load and preprocess the image
    file = request.files["image"]
    img = Image.open(file.stream)
    #Extract features and generate caption
    feature = extract_features(img)
    caption = predict_caption(caption_model, tokenizer, max_length, feature)
    return jsonify({"caption": caption})

if __name__ == "__main__":
    app.run(debug=True)
