import streamlit as st
import joblib
from preprocessing import preprocess_text

st.set_page_config(page_title="Spam Detector", page_icon="📩", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1000px;
    }

    h1, h2, h3 {
        color: #1f2937;
    }

    .custom-title {
        font-size: 42px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 0.2rem;
    }

    .custom-subtitle {
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #ffffff, #f3f4f6);
        padding: 20px;
        border-radius: 18px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        text-align: center;
        border: 1px solid #e5e7eb;
    }

    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #2563eb;
    }

    .metric-label {
        font-size: 15px;
        color: #6b7280;
        margin-top: 6px;
    }

    .result-spam {
        background: #fee2e2;
        color: #991b1b;
        padding: 18px;
        border-radius: 14px;
        font-size: 20px;
        font-weight: bold;
        border-left: 6px solid #dc2626;
    }

    .result-ham {
        background: #dcfce7;
        color: #166534;
        padding: 18px;
        border-radius: 14px;
        font-size: 20px;
        font-weight: bold;
        border-left: 6px solid #16a34a;
    }

    .clean-box {
        background: #111827;
        color: #f9fafb;
        padding: 16px;
        border-radius: 12px;
        font-family: monospace;
        font-size: 15px;
        overflow-wrap: break-word;
    }

    .stButton > button {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.5rem;
        font-size: 16px;
        font-weight: 600;
        transition: 0.3s;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8, #1e40af);
        transform: scale(1.02);
    }

    .stTextArea textarea {
        border-radius: 12px !important;
        border: 1px solid #d1d5db !important;
    }

    div[data-baseweb="select"] > div {
        border-radius: 12px !important;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    vectorizer = joblib.load("saved_models/vectorizer.pkl")
    nb_model = joblib.load("saved_models/naive_bayes.pkl")
    lr_model = joblib.load("saved_models/logistic_regression.pkl")
    mlp_model = joblib.load("saved_models/neural_network.pkl")
    return vectorizer, nb_model, lr_model, mlp_model


vectorizer, nb_model, lr_model, mlp_model = load_models()

models = {
    "Naive Bayes": nb_model,
    "Logistic Regression": lr_model,
    "Neural Network (MLP)": mlp_model
}

model_metrics = {
    "Naive Bayes": {
        "Accuracy": 0.9638,
        "Precision": 0.9919,
        "Recall": 0.7531,
        "F1-score": 0.8561
    },
    "Logistic Regression": {
        "Accuracy": 0.9753,
        "Precision": 0.8988,
        "Recall": 0.9321,
        "F1-score": 0.9152
    },
    "Neural Network (MLP)": {
        "Accuracy": 0.9788,
        "Precision": 0.9662,
        "Recall": 0.8827,
        "F1-score": 0.9226
    }
}


def predict_message(message, model, vectorizer):
    cleaned = preprocess_text(message)
    message_tfidf = vectorizer.transform([cleaned])
    prediction = model.predict(message_tfidf)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(message_tfidf)[0]
        confidence = max(proba)

    return prediction, confidence, cleaned


st.markdown(
    '<div class="custom-title">📩 Application intelligente de détection de Spam</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="custom-subtitle">Choisis un modèle de machine learning, saisis un message, puis vérifie automatiquement s’il s’agit d’un Spam ou d’un Ham.</div>',
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown("## ⚙️ Paramètres")
    selected_model_name = st.selectbox(
        "Choisir un modèle",
        list(models.keys())
    )

    st.markdown("---")
    st.markdown("### 📌 Modèles disponibles")
    st.write("- Naive Bayes")
    st.write("- Logistic Regression")
    st.write("- Neural Network (MLP)")

    st.markdown("---")
    st.markdown("### 🧠 Conseil")
    st.info("Le MLP est le meilleur modèle global selon les résultats obtenus.")

selected_model = models[selected_model_name]
metrics = model_metrics[selected_model_name]

st.markdown("## 📊 Performances du modèle choisi")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['Accuracy']:.4f}</div>
            <div class="metric-label">Accuracy</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['Precision']:.4f}</div>
            <div class="metric-label">Precision</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['Recall']:.4f}</div>
            <div class="metric-label">Recall</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['F1-score']:.4f}</div>
            <div class="metric-label">F1-score</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("## ✍️ Entrer un message")
message = st.text_area(
    "",
    height=180,
    placeholder="Exemple : Congratulations! You won a free iPhone. Click here now!"
)

if st.button("Prédire"):
    if not message.strip():
        st.warning("Veuillez entrer un message.")
    else:
        prediction, confidence, cleaned = predict_message(
            message, selected_model, vectorizer
        )

        st.markdown("## 🧹 Message après prétraitement")
        st.markdown(f'<div class="clean-box">{cleaned}</div>', unsafe_allow_html=True)

        st.markdown("## ✅ Résultat de la prédiction")

        if prediction == 1:
            st.markdown(
                '<div class="result-spam">🔴 SPAM détecté</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-ham">🟢 HAM détecté (message normal)</div>',
                unsafe_allow_html=True
            )

        if confidence is not None:
            st.progress(float(confidence))
            st.info(f"Niveau de confiance : {confidence:.2%}")

st.markdown("---")
st.markdown("## 🧪 Exemples de messages à tester")

example_col1, example_col2 = st.columns(2)

with example_col1:
    st.code("Congratulations! You won a free iPhone. Click here now!")
    st.code("URGENT! Your account has been suspended. Verify now!")

with example_col2:
    st.code("Hey, are we still meeting tonight?")
    st.code("You have won $10,000 cash! Claim now.")