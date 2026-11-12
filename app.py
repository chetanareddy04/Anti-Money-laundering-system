
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from keras.saving import load_model

# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="AML Detection System",
    page_icon="🏦",
    layout="wide"
)

BASE_DIR = Path(__file__).parent


# -----------------------------------
# CUSTOM CSS - ORIGINAL AML DESIGN
# -----------------------------------

st.markdown("""
<style>

/* PAGE LAYOUT */
.block-container {
    max-width: 100% !important;
    padding-top: 2rem !important;
    padding-bottom: 1rem !important;
    padding-left: 4% !important;
    padding-right: 4% !important;
}

/* TITLE BOX */
.title-box {
    width: 100%;
    min-height: 76px;
    box-sizing: border-box;
    display: flex;
    align-items: center;
    justify-content: center;

    background: linear-gradient(
        120deg,
        #0b1f3a,
        #123e68,
        #087f8c
    );

    border: 1px solid #d6ad60;
    border-left: 6px solid #e5b85c;
    border-radius: 12px;

    padding: 16px 20px;
    margin-top: 4px;
    margin-bottom: 8px;

    box-shadow: 0 4px 12px rgba(10, 35, 65, 0.15);
    overflow: visible;
}

/* TITLE TEXT */
.title-text {
    color: white !important;
    font-size: clamp(18px, 2vw, 25px);
    font-weight: 750;
    line-height: 1.4;
    text-align: center;
    margin: 0 !important;
    padding: 2px 0 !important;
}

/* DESCRIPTION */
.description {
    color: #64748b !important;
    font-size: 12px;
    line-height: 1.4;
    text-align: center;
    margin: 3px 0 14px 0 !important;
}

/* FIELD LABELS */
.field-label {
    display: block;
    font-size: 12px;
    font-weight: 700;
    line-height: 1.4;
    margin: 0 0 5px 0 !important;
    padding: 1px 0;
}

.blue-label {
    color: #1766a5 !important;
}

.teal-label {
    color: #008b83 !important;
}

.purple-label {
    color: #7652a3 !important;
}

.gold-label {
    color: #a87924 !important;
}

/* INPUT FIELDS */
div[data-testid="stSelectbox"] > label,
div[data-testid="stNumberInput"] > label {
    display: none;
}

div[data-baseweb="select"] > div {
    min-height: 38px;
    border-radius: 7px;
    border: 1px solid #cbd5e1;
}

div[data-testid="stNumberInput"] input {
    min-height: 38px;
    border-radius: 7px;
}

/* IMAGE */
div[data-testid="stImage"] img {
    width: 100%;
    max-height: 370px;
    object-fit: contain;
    border-radius: 12px;
}

/* BUTTON */
.stButton > button {
    width: 100%;
    height: 40px;
    min-height: 40px;

    background: linear-gradient(
        90deg,
        #123e68,
        #087f8c
    );

    color: white;
    border: none;
    border-radius: 8px;

    font-size: 13px;
    font-weight: 700;
    margin-top: 8px;
}

.stButton > button:hover {
    color: white;
    border: none;
    background: linear-gradient(
        90deg,
        #0b2d50,
        #066b75
    );
}

/* FOOTER */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# TITLE AND DESCRIPTION
# -----------------------------------

st.markdown("""
<div class="title-box">
    <div class="title-text">
        🛡️ Intelligent Anti-Money Laundering Detection System
    </div>
</div>


""", unsafe_allow_html=True)


# -----------------------------------
# LOAD MODEL AND PREPROCESSOR
# -----------------------------------

@st.cache_resource
def load_artifacts():

    model = load_model(
        BASE_DIR / "AML.keras",
        compile=False
    )

    preprocessor = joblib.load(
        BASE_DIR / "preprocessor"
    )

    return model, preprocessor


try:
    model, preprocessor = load_artifacts()

except Exception as e:
    st.error(f"Unable to load model or preprocessor: {e}")
    st.stop()


# -----------------------------------
# MAIN LAYOUT
# -----------------------------------

input_col, image_col = st.columns(
    [1.65, 1],
    gap="large"
)


# -----------------------------------
# INPUT FIELDS
# -----------------------------------

with input_col:

    # Transaction Amount
    st.markdown(
        '<div class="field-label gold-label">💰 Transaction Amount</div>',
        unsafe_allow_html=True
    )

    AMOUNT = st.number_input(
        "Amount",
        min_value=1.0,
        value=1000.0,
        step=100.0,
        label_visibility="collapsed"
    )

    # Currency fields
    currency_col1, currency_col2 = st.columns(2)

    with currency_col1:

        st.markdown(
            '<div class="field-label blue-label">💱 Payment Currency</div>',
            unsafe_allow_html=True
        )

        Payment_Currency = st.selectbox(
            "Payment Currency",
            [
                'UK pounds', 'Dirham', 'Albanian lek',
                'Euro', 'Swiss franc', 'Moroccan dirham',
                'Mexican Peso', 'Indian rupee', 'Naira',
                'Turkish lira', 'Pakistani rupee',
                'US dollar', 'Yen'
            ],
            label_visibility="collapsed"
        )

    with currency_col2:

        st.markdown(
            '<div class="field-label teal-label">💱 Received Currency</div>',
            unsafe_allow_html=True
        )

        Received_Currency = st.selectbox(
            "Received Currency",
            [
                'UK pounds', 'Pakistani rupee',
                'Moroccan dirham', 'Swiss franc',
                'Euro', 'Albanian lek', 'Indian rupee',
                'Yen', 'Turkish lira', 'Mexican Peso',
                'US dollar', 'Naira', 'Dirham'
            ],
            label_visibility="collapsed"
        )

    # Bank location fields
    location_col1, location_col2 = st.columns(2)

    with location_col1:

        st.markdown(
            '<div class="field-label purple-label">🏦 Sender Bank Location</div>',
            unsafe_allow_html=True
        )

        Sender_bank_location = st.selectbox(
            "Sender Bank Location",
            [
                'UK', 'Albania', 'Netherlands', 'Morocco',
                'Germany', 'Mexico', 'Switzerland', 'Spain',
                'Italy', 'India', 'Nigeria', 'France',
                'UAE', 'Turkey', 'Japan', 'Pakistan',
                'USA', 'Austria'
            ],
            label_visibility="collapsed"
        )

    with location_col2:

        st.markdown(
            '<div class="field-label blue-label">🏦 Receiver Bank Location</div>',
            unsafe_allow_html=True
        )

        Receiver_bank_location = st.selectbox(
            "Receiver Bank Location",
            [
                'UK', 'Pakistan', 'Morocco', 'Switzerland',
                'France', 'India', 'Spain', 'Turkey',
                'Mexico', 'USA', 'Austria', 'Nigeria',
                'Germany', 'Netherlands', 'UAE',
                'Albania', 'Japan', 'Italy'
            ],
            label_visibility="collapsed"
        )

    # Payment type
    st.markdown(
        '<div class="field-label teal-label">💳 Payment Type</div>',
        unsafe_allow_html=True
    )

    Payment_type = st.selectbox(
        "Payment Type",
        [
            'Cash Withdrawal',
            'ACH',
            'Cash Deposit',
            'Cheque',
            'Debit card',
            'Cross-border',
            'Credit card'
        ],
        label_visibility="collapsed"
    )

    # Prediction button
    predict = st.button(
        "🔍 Check for Money Laundering",
        use_container_width=True
    )


# -----------------------------------
# RIGHT SIDE IMAGE
# -----------------------------------

with image_col:

    image_path = BASE_DIR / "aml_security.jpeg"

    if image_path.exists():

        st.image(
            str(image_path),
            use_container_width=True
        )

    else:

        st.warning(
            "Image not found. Check the filename: aml_security.jpeg"
        )


# -----------------------------------
# PREDICTION
# -----------------------------------

if predict:

    input_data = pd.DataFrame({

        'Amount': [AMOUNT],

        'Payment_currency': [Payment_Currency],

        'Received_currency': [Received_Currency],

        'Sender_bank_location': [Sender_bank_location],

        'Receiver_bank_location': [Receiver_bank_location],

        'Payment_type': [Payment_type]

    })

    try:

        input_transformed = preprocessor.transform(
            input_data
        )

        prediction = model.predict(
            input_transformed,
            verbose=0
        )

        probability = float(
            np.asarray(prediction).flatten()[0]
        )

        if probability > 0.5:

            st.error("⚠️ Money Laundering Detected!")

            st.warning(
                "This transaction is identified as suspicious "
                "and may be associated with money laundering. "
                "Further investigation is recommended."
            )

        else:

            st.success("✅ No Money Laundering Detected!")

            st.info(
                "This transaction is classified as legitimate "
                "by the model."
            )

    except Exception as e:

        st.error(f"Prediction error: {e}")
