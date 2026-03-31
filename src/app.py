import streamlit as st
import pandas as pd
import joblib

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Laptop Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# CUSTOM CSS FOR PROFESSIONAL STYLE
# -------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to bottom, #e6f0ff, #f0f2f6);
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3 {
        color: #0f3d91;
    }
    .card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .card h4 {
        margin: 0;
        color: #0f3d91;
    }
    .card p {
        font-size: 18px;
        font-weight: bold;
        margin: 5px 0 0 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# LOAD MODEL
# -------------------------
bundle = joblib.load("models/laptop_price_model.pkl")
st.write(bundle.keys())
models = bundle["models"]
features = bundle["features"]
metrics = bundle["metrics"]

brand_cols = [col for col in features if col.startswith("Brand_")]
brand_names = [col.replace("Brand_", "") for col in brand_cols]

st.title("💻 Laptop Price Prediction Dashboard")

# -------------------------
# SIDEBAR INPUTS
# -------------------------
st.sidebar.header("Laptop 1")
brand1 = st.sidebar.selectbox("Brand", brand_names, key="b1")
inches1 = st.sidebar.slider("Screen Size (Inches)", 11.0, 18.0, 15.6, key="i1")
ram1 = st.sidebar.selectbox("RAM (GB)", [4, 8, 16, 32], key="r1")
weight1 = st.sidebar.slider("Weight (kg)", 1.0, 3.5, 2.0, key="w1")
cpu1 = st.sidebar.slider("CPU Frequency (GHz)", 1.0, 5.0, 2.5, key="c1")
storage1 = st.sidebar.selectbox("Total Storage (GB)", [128, 256, 512, 1024, 2048], key="s1")

st.sidebar.header("Laptop 2")
brand2 = st.sidebar.selectbox("Brand", brand_names, key="b2")
inches2 = st.sidebar.slider("Screen Size (Inches)", 11.0, 18.0, 15.6, key="i2")
ram2 = st.sidebar.selectbox("RAM (GB)", [4, 8, 16, 32], key="r2")
weight2 = st.sidebar.slider("Weight (kg)", 1.0, 3.5, 2.0, key="w2")
cpu2 = st.sidebar.slider("CPU Frequency (GHz)", 1.0, 5.0, 2.5, key="c2")
storage2 = st.sidebar.selectbox("Total Storage (GB)", [128, 256, 512, 1024, 2048], key="s2")

# -------------------------
# PREDICT BUTTON
# -------------------------
if st.button("Predict Prices"):

    def prepare_input(inches, ram, weight, cpu, storage, brand):
        base = pd.DataFrame([[inches, ram, weight, cpu, storage]],
                            columns=[f for f in features if not f.startswith("Brand_")])
        brand_df = pd.DataFrame([[1 if b == brand else 0 for b in brand_names]], columns=brand_cols)
        return pd.concat([base, brand_df], axis=1)

    input1 = prepare_input(inches1, ram1, weight1, cpu1, storage1, brand1)
    input2 = prepare_input(inches2, ram2, weight2, cpu2, storage2, brand2)

    st.subheader("💰 Laptop Price Predictions (Side by Side)")

    # Create two columns for cards
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"<div class='card'><h4>Laptop 1</h4></div>", unsafe_allow_html=True)
        for name, model in models.items():
            price = int(model.predict(input1)[0])
            st.markdown(f"<div class='card'><p>{name}: ₹ {price:,}</p></div>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"<div class='card'><h4>Laptop 2</h4></div>", unsafe_allow_html=True)
        for name, model in models.items():
            price = int(model.predict(input2)[0])
            st.markdown(f"<div class='card'><p>{name}: ₹ {price:,}</p></div>", unsafe_allow_html=True)

    # Comparison bar chart
    comparison = pd.DataFrame(
        [[int(models[m].predict(input1)[0]), int(models[m].predict(input2)[0])] for m in models],
        index=models.keys(),
        columns=["Laptop 1", "Laptop 2"]
    )
    st.bar_chart(comparison)

    # -------------------------
    # METRICS SECTION
    # -------------------------
    st.subheader("📊 Model Metrics on Test Data")
    for name, m in metrics.items():
        st.markdown(f"<div class='card'><p>{name} → R²: {m['r2']:.2f}, MAE: ₹ {int(m['mae']):,}</p></div>", unsafe_allow_html=True)
