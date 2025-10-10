from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import numpy as np
import pandas
import sklearn
import pickle
import cv2
import pytesseract
from PIL import Image
import re

# Configure Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load ML models
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
mx = pickle.load(open('minmaxscaler.pkl', 'rb'))

app = Flask(__name__)
CORS(app)

# ============================================
# HTML TEMPLATE ROUTE
# ============================================
@app.route('/')
def index():
    return render_template("index.html")

# ============================================
# CROP RECOMMENDATION API
# ============================================
@app.route("/predict", methods=['POST'])
def predict():
    try:
        if request.is_json:
            data = request.json
            N = float(data['Nitrogen'])
            P = float(data['Phosporus'])
            K = float(data['Potassium'])
            temp = float(data['Temperature'])
            humidity = float(data['Humidity'])
            ph = float(data['pH'])
            rainfall = float(data['Rainfall'])
        else:
            N = request.form['Nitrogen']
            P = request.form['Phosporus']
            K = request.form['Potassium']
            temp = request.form['Temperature']
            humidity = request.form['Humidity']
            ph = request.form['pH']
            rainfall = request.form['Rainfall']

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        mx_features = mx.transform(single_pred)
        sc_mx_features = sc.transform(mx_features)
        prediction = model.predict(sc_mx_features)

        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 
            6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 
            11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 
            16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 
            20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
        }

        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            
            if request.is_json:
                return jsonify({
                    "crop": crop,
                    "message": "is the best crop to be cultivated right there"
                })
            else:
                result = "{} is the best crop to be cultivated right there".format(crop)
                return render_template('index.html', result=result)
        else:
            if request.is_json:
                return jsonify({"error": "Could not determine the best crop"}), 400
            else:
                result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
                return render_template('index.html', result=result)
    
    except Exception as e:
        if request.is_json:
            return jsonify({"error": str(e)}), 500
        else:
            return render_template('index.html', result="Error: " + str(e))

# ============================================
# OCR API - CROPS IMAGE TO READ TOP SECTION ONLY
# ============================================
@app.route("/ocr/soil-card", methods=['POST'])
def ocr_soil_card():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        image = Image.open(file.stream)
        
        # Convert to OpenCV format
        img_array = np.array(image)
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Resize if too large
        height, width = img.shape[:2]
        if width > 1500:
            scale = 1500 / width
            img = cv2.resize(img, None, fx=scale, fy=scale)
            height, width = img.shape[:2]
        
        # ⭐ CROP TO TOP 40% OF IMAGE ONLY (where soil test values are)
        crop_height = int(height * 0.4)  # Only read top 40%
        img = img[0:crop_height, 0:width]
        
        print(f"🖼️  Original size: {width}x{height}, Cropped to: {width}x{crop_height}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Multiple preprocessing methods
        _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        _, thresh3 = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Try OCR with all methods
        results = []
        for i, processed_img in enumerate([gray, thresh1, thresh2, thresh3]):
            custom_config = r'--oem 3 --psm 6 -l eng'
            text = pytesseract.image_to_string(processed_img, config=custom_config)
            extracted = extract_soil_data(text)
            
            # Count found values
            found_count = sum(1 for v in extracted.values() if v is not None)
            results.append((found_count, text, extracted))
            
            print(f"\n🔍 Method {i+1} found {found_count}/6 values")
        
        # Use best result
        results.sort(reverse=True, key=lambda x: x[0])
        best_result = results[0]
        
        return jsonify({
            "success": True,
            "rawText": best_result[1],
            "extractedData": best_result[2]
        })
            
    except Exception as e:
        print(f"❌ Error in OCR: {str(e)}")
        return jsonify({"error": str(e)}), 500


def extract_soil_data(text):
    """Extract NPK, pH, EC, OC using symbols - Works with Kannada cards"""
    data = {
        "nitrogen": None,
        "phosphorus": None,
        "potassium": None,
        "ph": None,
        "ec": None,
        "oc": None
    }
    
    print("\n" + "=" * 60)
    print("📄 RAW OCR TEXT:")
    print(text)
    print("=" * 60)
    
    # Clean text
    all_text = ' '.join(text.split())
    
    # Extract all numbers (including decimals)
    all_numbers = re.findall(r'\d+\.?\d*', text)
    print(f"🔢 All numbers found: {all_numbers}")
    
    # ========== NITROGEN ==========
    n_patterns = [
        r'\(N\)\s*(\d+\.?\d*)',
        r'N\s*\)\s*(\d+\.?\d*)',
        r'N\s+(\d+\.?\d*)',
        r'nitrogen.*?(\d+\.?\d*)',
    ]
    for pattern in n_patterns:
        n_match = re.search(pattern, text, re.IGNORECASE)
        if n_match:
            val = float(n_match.group(1))
            if 50 < val < 600:
                data["nitrogen"] = val
                print(f"✅ Nitrogen (N): {val} kg/ha")
                break
    
    # ========== PHOSPHORUS ==========
    p_patterns = [
        r'\(P\)\s*(\d+\.?\d*)',
        r'P\s*\)\s*(\d+\.?\d*)',
        r'P\s+(\d+\.?\d*)',
        r'phosphorus.*?(\d+\.?\d*)',
    ]
    for pattern in p_patterns:
        p_match = re.search(pattern, text, re.IGNORECASE)
        if p_match:
            val = float(p_match.group(1))
            if 0.5 < val < 100:
                data["phosphorus"] = val
                print(f"✅ Phosphorus (P): {val} kg/ha")
                break
    
    # ========== POTASSIUM ==========
    k_patterns = [
        r'\(K\)\s*(\d+\.?\d*)',
        r'K\s*\)\s*(\d+\.?\d*)',
        r'K\s+(\d+\.?\d*)',
        r'potassium.*?(\d+\.?\d*)',
    ]
    for pattern in k_patterns:
        k_match = re.search(pattern, text, re.IGNORECASE)
        if k_match:
            val = float(k_match.group(1))
            if 50 < val < 800:
                data["potassium"] = val
                print(f"✅ Potassium (K): {val} kg/ha")
                break
    
    # ========== pH ==========
    ph_patterns = [
        r'pH\s*[:\-]?\s*(\d+\.?\d*)',
        r'ph\s*[:\-]?\s*(\d+\.?\d*)',
        r'PH\s*[:\-]?\s*(\d+\.?\d*)',
    ]
    for pattern in ph_patterns:
        ph_match = re.search(pattern, text, re.IGNORECASE)
        if ph_match:
            val = float(ph_match.group(1))
            if 3 < val < 10:
                data["ph"] = val
                print(f"✅ pH: {val}")
                break
    
    # ========== EC (Electrical Conductivity) ==========
    ec_patterns = [
        r'EC\s*[:\-]?\s*(\d+\.?\d*)',
        r'ec\s*[:\-]?\s*(\d+\.?\d*)',
        r'\(EC\)\s*(\d+\.?\d*)',
    ]
    for pattern in ec_patterns:
        ec_match = re.search(pattern, text, re.IGNORECASE)
        if ec_match:
            val = float(ec_match.group(1))
            if 0 < val < 5:
                data["ec"] = val
                print(f"✅ EC: {val} dS/m")
                break
    
    # ========== OC (Organic Carbon) ==========
    oc_patterns = [
        r'OC\s*[:\-]?\s*(\d+\.?\d*)',
        r'oc\s*[:\-]?\s*(\d+\.?\d*)',
        r'\(OC\)\s*(\d+\.?\d*)',
        r'organic.*?carbon.*?(\d+\.?\d*)',
    ]
    for pattern in oc_patterns:
        oc_match = re.search(pattern, text, re.IGNORECASE)
        if oc_match:
            val = float(oc_match.group(1))
            if 0 < val < 5:
                data["oc"] = val
                print(f"✅ OC: {val}%")
                break
    
    # ========== FALLBACK LOGIC ==========
    numbers = [float(n) for n in all_numbers if '.' in n or len(n) > 1]
    numbers = sorted(set(numbers), reverse=True)
    
    print(f"\n🔍 FALLBACK MODE - Detected numbers: {numbers}")
    
    # Fallback for NPK
    if not data["nitrogen"] or not data["phosphorus"] or not data["potassium"]:
        for num in numbers:
            if not data["nitrogen"] and 100 < num < 600:
                data["nitrogen"] = num
                print(f"⚠️  Fallback Nitrogen: {num} kg/ha")
            elif not data["potassium"] and 100 < num < 600 and num != data.get("nitrogen"):
                data["potassium"] = num
                print(f"⚠️  Fallback Potassium: {num} kg/ha")
            elif not data["phosphorus"] and 1 < num < 50:
                data["phosphorus"] = num
                print(f"⚠️  Fallback Phosphorus: {num} kg/ha")
    
    # Fallback for pH (typically 5-8)
    if not data["ph"]:
        for num in numbers:
            if 4 < num < 9:
                data["ph"] = num
                print(f"⚠️  Fallback pH: {num}")
                break
    
    # Fallback for EC (typically 0.1-2.0 dS/m)
    if not data["ec"]:
        for num in numbers:
            if 0.05 < num < 3.0 and num < 10:
                # Make sure it's not pH or OC
                if num != data.get("ph") and num != data.get("oc"):
                    data["ec"] = num
                    print(f"⚠️  Fallback EC: {num} dS/m")
                    break
    
    # Fallback for OC (typically 0.5-3.0%)
    if not data["oc"]:
        for num in numbers:
            if 0.3 < num < 4.0 and num < 10:
                # Make sure it's not pH or EC
                if num != data.get("ph") and num != data.get("ec"):
                    data["oc"] = num
                    print(f"⚠️  Fallback OC: {num}%")
                    break
    
    print("=" * 60)
    print("📊 FINAL EXTRACTED DATA:")
    for key, value in data.items():
        if value is not None:
            print(f"   {key.upper()}: {value}")
        else:
            print(f"   {key.upper()}: Not found")
    print("=" * 60 + "\n")
    
    return data


if __name__ == "__main__":
    app.run(debug=True, port=5000)
