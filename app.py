%%writefile app.py
import streamlit as st
import cv2
import requests
import numpy as np
from pyzbar.pyzbar import decode
import google.generativeai as genai
from PIL import Image

# ---------------- CONFIG ----------------
st.set_page_config(page_title="FoodScan", layout="centered")

#  Replace with st.secrets["GEMINI_API_KEY"] in production
genai.configure(api_key="AIzaSyDNUi_YOiUFDOUF24HKOTk8qrbbMpCrqts")

# ---------------- FUNCTIONS ----------------

def scan_barcode(image):
    """
    Detect barcode from RGB image
    """
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    barcodes = decode(img)
    if not barcodes:
        return None, None
    barcode = barcodes[0]
    return barcode.data.decode("utf-8"), barcode

def get_product_from_api(barcode):
    url = f"https://world.openfoodfacts.net/api/v2/product/{barcode}"
    r = requests.get(url)
    if r.status_code != 200:
        return None

    data = r.json()
    if data.get("status") != 1:
        return None

    product = data.get("product", {})
    return {
        "name": product.get("product_name", "Unknown"),
        "nutriments": product.get("nutriments", {}),
        "ingredients": product.get("ingredients_text", ""),
        "labels": product.get("labels", "")
    }

def health_decision(user, product):
    prompt = f"""
You are a food health recommendation system.

User profile:
- Diabetes: {user['diabetes']}
- BP: {user['bp']}
- Heart disease: {user['heart']}
- Age: {user['age']}
- Diet: {user['diet']}

Food nutrition (per 100g):
- Sugar: {product['nutriments'].get('sugars_100g', 0)}
- Salt: {product['nutriments'].get('salt_100g', 0)}
- Saturated Fat: {product['nutriments'].get('saturated-fat_100g', 0)}

Give a clear recommendation:
- Recommended / Consume with caution / Not recommended
- Explain why in simple language.
"""
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text

# ---------------- UI ----------------

st.title(" FoodScan – Smart Food Analyzer")

# ---- Health Profile ----
st.subheader(" Health Profile")
age = st.number_input("Age", min_value=1, max_value=120, value=24)
diet = st.selectbox("Diet Type", ["Vegetarian", "Non-Vegetarian"])
diabetes = st.checkbox("Diabetes")
bp = st.checkbox("High Blood Pressure")
heart = st.checkbox("Heart Disease")

user_profile = {
    "age": age,
    "diet": diet,
    "diabetes": diabetes,
    "bp": bp,
    "heart": heart
}

st.divider()

# ---- Scan Mode ----
scan_mode = st.radio("Choose scan method", [" Upload Image", " Camera Scan"])

image = None

# ---- Upload Image ----
if scan_mode == " Upload Image":
    uploaded_file = st.file_uploader(
        "Upload barcode image",
        type=["jpg", "png", "jpeg", "webp"]
    )

    if uploaded_file:
        image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ---- Camera Scan ----
elif scan_mode == " Camera Scan":
    camera_image = st.camera_input("Scan barcode using camera")

    if camera_image is not None:
        pil_image = Image.open(camera_image)
        image = np.array(pil_image)

# ---- Process Image ----
if image is not None:
    if image.ndim not in [2, 3]:
        st.error("Invalid image format")
        st.stop()

    st.image(image, caption="Input Image", use_column_width=True)

    barcode_data, barcode_obj = scan_barcode(image)

    if barcode_data:
        st.success(f" Barcode detected: {barcode_data}")

        # draw bounding box
        x, y, w, h = barcode_obj.rect
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        st.image(image, caption="Detected Barcode", use_column_width=True)

        product = get_product_from_api(barcode_data)

        if product:
            st.subheader(" Product Info")
            st.write("**Name:**", product["name"])
            st.write("**Ingredients:**", product["ingredients"])

            st.subheader(" Health Recommendation")
            with st.spinner("Analyzing with AI..."):
                result = health_decision(user_profile, product)

            st.info(result)
        else:
            st.error(" Product not found in OpenFoodFacts")
    else:
        st.error(" No barcode detected. Try better lighting or angle.")