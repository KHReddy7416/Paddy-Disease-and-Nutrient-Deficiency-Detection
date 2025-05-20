from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from io import BytesIO
import base64

app = Flask(__name__)

# Load Models
disease_model_path = os.path.join(os.path.dirname(__file__), "model", "Dis_Model.keras")
deficiency_model_path = os.path.join(os.path.dirname(__file__), "model", "Def_Model.keras")

disease_model = tf.keras.models.load_model(disease_model_path)
deficiency_model = tf.keras.models.load_model(deficiency_model_path)

# Class Labels
disease_classes = ['Bacterial_Leaf_Blight', 'Bacterial_Leaf_Streak', 'Bacterial_Panicle_Blight',
                   'Blast', 'Brown_Spot', 'Dead_Heart', 'Downy_Mildew', 'False_Smut',
                   'Hispa', 'Normal', 'Tungro']

deficiency_classes = ['Nitrogen_Deficiency', 'Normal', 'Phosphorus_Deficiency', 'Potassium_Deficiency']

# Precautions (unchanged)
precautions = {
    "Bacterial_Leaf_Blight": (
        ["Use resistant varieties.", "Avoid excessive nitrogen.", "Ensure proper field sanitation.", "Use copper-based fungicides.", "Monitor fields regularly."],
        ["నిరోధక రకాలన్ని వాడండి.", "అధిక నత్రజని వాడకండి.", "తగిన పంట పరిశుభ్రతను పాటించండి.", "కాపర్ ఆధారిత భాక్తునాలు వాడండి.", "క్షేత్రాన్ని క్రమం తప్పకుండా పరిశీలించండి."]
    ),
    "Bacterial_Leaf_Streak": (
        ["Improve drainage.", "Apply Copper fungicides.", "Use disease-free seeds.", "Avoid overcrowding of plants.", "Maintain balanced fertilization."],
        ["నిర్మల నికాసంని మెరుగుపరచండి.", "కాప్పర్ భాక్తునాలు వాడండి.", "వ్యాధిలౌని విత్తనాలు వాడండి.", "మొక్కలను ఎక్కువగా తాడకుండా నాటండి.", "సమతుల్య ఎరువుల ను వాడండి."]
    ),
    "Bacterial_Panicle_Blight": (
        ["Use certified seeds.", "Avoid dense planting.", "Proper irrigation management.", "Monitor panicle development.", "Apply appropriate fungicides."],
        ["ప్రామాణిక విత్తనాలు వాడండి.", "ఘనమైన నాటకాన్ని నివారించండి.", "నీటిపారుదల యంత్రణ సరిగ్గా వుండాలి.", "పానికిల్ అభివృద్ధిని పర్యవేక్షించండి.", "తగిన ఫంగిసైడ్స్ వాడండి."]
    ),
    "Blast": (
        ["Use resistant varieties.", "Spray Tricyclazole.", "Avoid high nitrogen levels.", "Ensure good air circulation.", "Maintain field sanitation."],
        ["నిరోధక రకాలన్ని వాడండి.", "ట్రైసైక్లోజోల్ స్ప్రే చేయండి.", "అధిక నత్రజని నివారించండి.", "గాలి ప్రసరణ బాగుండేలా చూసుకోండి.", "పరిశుభ్రతను పాటించండి."]
    ),
    "Brown_Spot": (
        ["Apply Potassium fertilizers.", "Use Mancozeb (Dithane M-45).", "Improve field drainage.", "Avoid water stagnation.", "Follow proper spacing."],
        ["పొటాషియం ఎరువులు వాడండి.", "మ్యాంకోజెబ్ వాడండి.", "నికాసాన్ని మెరుగుపరచండి.", "నీటి నిల్వ నివారించండి.", "తగినంతంతదూరం ఉంచండి."]
    ),
    "Dead_Heart": (
        ["Use light traps.", "Apply Chlorantraniliprole.", "Monitor stem borer activity.", "Remove affected tillers.", "Keep field weed-free."],
        ["లైట్ ట్రాప్ వాడండి.", "క్లోరాన్ట్రానిలిప్రోల్ స్ప్రే చేయండి.", "స్టెం బోరర్ క్రియాశీలతను పర్యవేక్షించండి.", "దెబ్బతిన్న మొక్కలను తీసేయండి.", "వేడి లేకుండా ఉంచండి."]
    ),
    "Downy_Mildew": (
        ["Ensure good airflow.", "Avoid excessive moisture.", "Use fungicides as recommended.", "Space plants properly.", "Use resistant varieties."],
        ["మంచి గాలి ప్రసరణ ఉండేలా చూసుకోండి.", "అధిక తేమ నివారించండి.", "సిఫారసు చేసిన ఫంగిసైడ్స్ వాడండి.", "తగిన స్పేసింగ్ పాటించండి.", "నిరోధక రకాలను వాడండి."]
    ),
    "False_Smut": (
        ["Apply Propiconazole at booting stage.", "Use clean seeds.", "Avoid late planting.", "Maintain field hygiene.", "Control weeds."],
        ["బూటింగ్ దశలో ప్రోపికోనజోల్ స్ప్రే చేయండి.", "శుభ్రమైన విత్తనాలు వాడండి.", "తరువాయి నాటకం నివారించండి.", "పరిశుభ్రత పాటించండి.", "వెడ్స్ నియంత్రించండి."]
    ),
    "Hispa": (
        ["Handpick larvae.", "Apply Imidacloprid.", "Avoid excessive nitrogen.", "Destroy pupae in leaf sheaths.", "Use pheromone traps."],
        ["లార్వాలను చేతితో తీసేయండి.", "ఇమిడాక్లోప్రిడ్ వాడండి.", "అధిక నత్రజని నివారించండి.", "ఆకు కప్పులలోని పుప్పాలను నాశనం చేయండి.", "ఫెరోమోన్ ట్రాప్స్ వాడండి."]
    ),
    "Normal": (
        ["No precautions needed.", "Maintain proper crop management.", "Regular monitoring is advised.", "Follow recommended farming practices.", "Ensure proper irrigation."],
        ["ఎలాంటి జాగ్రత్తలు అవసరం లేదు.", "పంట నిర్వహణను సరిగా చేయండి.", "క్రమం తప్పకుండా పర్యవేక్షణ చేయండి.", "సిఫారసైన వ్యవసాయ పద్ధతులను పాటించండి.", "తగిన నీటిపారుదల అందించండి."]
    ),
    "Tungro": (
        ["Control leafhoppers.", "Remove infected plants.", "Use resistant varieties.", "Avoid overlapping crops.", "Apply recommended insecticides."],
        ["లీఫ్ హాపర్లను నియంత్రించండి.", "సంవిధానమైన మొక్కలను తొలగించండి.", "నిరోధక రకాలన్ని వాడండి.", "ఓవర్‌ల్యాపింగ్ పంటలను నివారించండి.", "సిఫారసు చేసిన పురుగుమందులు వాడండి."]
    ),
    "Nitrogen_Deficiency": (
        ["Apply urea or ammonium sulfate fertilizers. Use compost or organic matter to enrich the soil."],
        ["యూరియా లేదా అమ్మోనియం సల్ఫేట్ ఎరువులు వేసండి. మట్టిని సమృద్ధిగా చేసేందుకు కంపోస్ట్ లేదా సేంద్రీయ పదార్థాలు వాడండి."]
    ),
    "Normal_Deficiency": (
        ["No deficiency detected. Maintain current nutrient practices."],
        ["యే లోపం కనబడలేదు. ప్రస్తుత పోషక విధానాలను కొనసాగించండి."]
    ),
    "Phosphorus_Deficiency": (
        ["Use single super phosphate (SSP) or diammonium phosphate (DAP). Avoid overwatering."],
        ["సింగిల్ సూపర్ ఫాస్ఫేట్ (SSP) లేదా డైఅమ్మోనియం ఫాస్ఫేట్ (DAP) వాడండి. ఎక్కువ నీరు వేయడం మానండి."]
    ),
    "Potassium_Deficiency": (
        ["Apply muriate of potash (MOP) or use potassium-rich organic fertilizers."],
        ["మ్యూరియేట్ ఆఫ్ పొటాష్ (MOP) లేదా పొటాష్ ఎక్కువగా ఉన్న సేంద్రియ ఎరువులు వాడండి."]
    )

}

# Image Preprocessing
def preprocess(image):
    image = Image.open(image).convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# Ensure the uploads directory exists
uploads_dir = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(uploads_dir, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files or "detection_type" not in request.form:
        return "Missing image or detection type.", 400

    file = request.files["image"]
    detection_type = request.form["detection_type"]

    if file.filename == "":
        return "No file selected.", 400

    try:
        image_array = preprocess(file)
    except Exception as e:
        return f"Image processing failed: {e}", 500

    if detection_type == "disease":
        model = disease_model
        class_names = disease_classes
    elif detection_type == "deficiency":
        model = deficiency_model
        class_names = deficiency_classes
    else:
        return "Invalid detection type.", 400

    prediction = model.predict(image_array)
    class_idx = np.argmax(prediction)
    confidence = round(float(np.max(prediction)) * 100, 2)
    predicted_class = class_names[class_idx]

    if predicted_class == "Normal":
        predicted_class_key = "Normal" if detection_type == "disease" else "Normal_Deficiency"
    else:
        predicted_class_key = predicted_class

    precaution_en, precaution_te = precautions.get(predicted_class_key, ([], []))


        # Save the image
    image_path = os.path.join(uploads_dir, file.filename)
    file.seek(0)
    file.save(image_path)

    # Convert to base64 for HTML display
    img = Image.open(image_path).convert("RGB")
    image_bytes = BytesIO()

    # Try to use the original image format if available and supported, otherwise use JPEG
    try:
        original_format = img.format if img.format else "JPEG"
        if original_format.upper() not in ["JPEG", "JPG", "PNG", "BMP", "GIF", "TIFF"]:
            original_format = "JPEG"  # fallback if unsupported
    except:
        original_format = "JPEG"

    img.save(image_bytes, format=original_format)
    image_base64 = base64.b64encode(image_bytes.getvalue()).decode('utf-8')

 

    return render_template(
        "result.html",
        predicted_class=predicted_class,
        confidence=confidence,
        precautions_en=precaution_en,
        precautions_te=precaution_te,
        image_base64=image_base64
    )

if __name__ == "__main__":
    app.run(debug=True)



