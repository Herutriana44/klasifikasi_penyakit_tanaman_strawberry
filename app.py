"""
Flask Web App - Klasifikasi Penyakit Tanaman Strawberry dengan GoogleNet
"""
import os
import json
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import torch
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global model & classes
model = None
class_names = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transform (harus sama dengan training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_model():
    """Load model dan class names"""
    global model, class_names
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'googlenet_model.pth')
    classes_path = os.path.join(os.path.dirname(__file__), 'models', 'class_names.json')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    num_classes = checkpoint.get('num_classes', len(checkpoint.get('class_names', [])))
    class_names = checkpoint.get('class_names', [])

    if not class_names and os.path.exists(classes_path):
        with open(classes_path) as f:
            class_names = json.load(f)

    model = models.googlenet(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(image_path):
    """Prediksi kelas dari gambar"""
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)

    probs = torch.softmax(output.logits if hasattr(output, 'logits') else output, dim=1)
    prob, idx = torch.max(probs, 1)
    idx = idx.item()
    prob = prob.item()

    return {
        'class': class_names[idx] if idx < len(class_names) else f'Class_{idx}',
        'confidence': round(prob * 100, 2),
        'all_probs': {class_names[i]: round(probs[0][i].item() * 100, 2) for i in range(len(class_names))}
    }


@app.route('/')
def index():
    return render_template('index.html', classes=class_names)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files and 'image' not in request.files:
        return jsonify({'error': 'Tidak ada file gambar'}), 400

    file = request.files.get('file') or request.files.get('image')
    if file.filename == '':
        return jsonify({'error': 'File tidak dipilih'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Format tidak didukung. Gunakan: png, jpg, jpeg, gif, webp'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        result = predict_image(filepath)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})


if __name__ == '__main__':
    load_model()
    print("Model loaded. Classes:", class_names)
    app.run(host='0.0.0.0', port=5000, debug=False)
