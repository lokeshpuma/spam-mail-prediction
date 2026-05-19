import streamlit as st
import pickle
import io
import os

st.set_page_config(page_title="Spam Mail Detector", layout="centered")

@st.cache_resource
def load_model(path="spam_mail_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_vectorizer_from_file(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_vectorizer_from_bytes(b):
    return pickle.load(io.BytesIO(b))

model = None
try:
    model = load_model()
except Exception as e:
    st.warning("Failed to load `spam_mail_model.pkl` automatically — you can still upload a vectorizer below.")

# model feature diagnostics
model_n_features = None
model_feature_names = None
if model is not None:
    model_n_features = getattr(model, "n_features_in_", None)
    model_feature_names = getattr(model, "feature_names_in_", None)

st.title("Spam Mail Detector")
st.write("Enter the email text below and press Predict. If your model is a raw estimator, upload its vectorizer (pickle) in the sidebar.")

sample = """Hi there,

You won a prize! Click the link to claim your reward.

Best,
Scammy"""

text = st.text_area("Email text", value=sample, height=250)

st.sidebar.header("Model / Vectorizer")
st.sidebar.write("Model file: spam_mail_model.pkl")

# Allow user to upload a vectorizer pickle (optional)
uploaded_vect = st.sidebar.file_uploader("Upload vectorizer pickle (optional)", type=["pkl", "pickle"])
vectorizer = None
if uploaded_vect is not None:
    try:
        vectorizer = load_vectorizer_from_bytes(uploaded_vect.read())
        st.sidebar.success("Vectorizer uploaded and loaded")
    except Exception as e:
        st.sidebar.error("Failed to load uploaded vectorizer")
        st.sidebar.exception(e)
else:
    # try common filename
    for fname in ("vectorizer.pkl", "tfidf.pkl", "count_vectorizer.pkl"):
        if os.path.exists(fname):
            try:
                vectorizer = load_vectorizer_from_file(fname)
                st.sidebar.write(f"Loaded vectorizer from {fname}")
                break
            except Exception:
                pass

if model is not None:
    try:
        st.sidebar.write(type(model))
        if model_n_features is not None:
            st.sidebar.write(f"Model expects {model_n_features} numeric features (model.n_features_in_)")
        if model_feature_names is not None:
            st.sidebar.write("Model feature names available")
            st.sidebar.write(list(model_feature_names[:20]))
        if hasattr(model, "named_steps"):
            st.sidebar.write("Pipeline steps:")
            st.sidebar.write(list(model.named_steps.keys()))
    except Exception:
        pass

st.sidebar.markdown("---")

if st.button("Predict"):
    if model is None:
        st.error("No model loaded: place `spam_mail_model.pkl` in the app directory.")
    else:
        # First attempt: if model looks like a pipeline or accepts raw text, try direct predict
        tried_transform = False
        try:
            pred = model.predict([text])
            label = pred[0]
            tried_transform = True
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba([text])[0]
                prob_spam = probs[1] if len(probs) > 1 else probs[0]
                st.success(f"Prediction: **{'Spam' if int(label) == 1 else 'Not Spam'}**")
                st.write(f"Spam probability: {prob_spam:.3f}")
            else:
                st.success(f"Prediction: **{'Spam' if int(label) == 1 else 'Not Spam'}**")
        except Exception as exc:
            # If failure indicates the estimator expects numeric features, try using vectorizer if available
            msg = str(exc)
            if ("Expected 2D array" in msg or "reshape" in msg or "n_features" in msg) and vectorizer is not None:
                try:
                    X = vectorizer.transform([text])
                    pred = model.predict(X)
                    label = pred[0]
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(X)[0]
                        prob_spam = probs[1] if len(probs) > 1 else probs[0]
                        st.success(f"Prediction: **{'Spam' if int(label) == 1 else 'Not Spam'}**")
                        st.write(f"Spam probability: {prob_spam:.3f}")
                    else:
                        st.success(f"Prediction: **{'Spam' if int(label) == 1 else 'Not Spam'}**")
                    tried_transform = True
                except Exception as e2:
                    st.error("Prediction with uploaded vectorizer failed.")
                    st.exception(e2)
            else:
                st.error("Prediction failed. The loaded model may require preprocessing or a vectorizer (e.g., a pipeline).")
                st.exception(exc)

        if not tried_transform:
            st.info("If the model is a bare estimator, upload its vectorizer pickle in the sidebar, or use a Pipeline combining vectorizer+estimator.")

