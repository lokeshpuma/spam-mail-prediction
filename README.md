# Spam Mail Detector

This project provides a minimal Streamlit web app to classify emails as spam or not-spam using a pickled scikit-learn model.

Files
- `spam_mail_model.pkl`: Trained model pickle (must be in the same directory as the app).
- `app.py`: Streamlit application.
- `requirements.txt`: Python dependencies.

Quick start

1. Activate your conda environment (you mentioned `tf`):

```bash
conda activate tf
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

Notes
- If the model requires text vectorization (e.g., `TfidfVectorizer`), the pickled object should be a scikit-learn `Pipeline` combining the vectorizer and estimator. If you only provide the raw estimator, the app will raise an error when attempting to predict raw text.
- The app will show model type and pipeline steps (if present) in the sidebar.

Vectorizer (when model is a bare estimator)

- If your `spam_mail_model.pkl` is a raw estimator (e.g., `LogisticRegression`) that was trained on vectorized features, you must provide the corresponding vectorizer so the app can transform raw text into the numeric features the model expects.
- You can either:
	- Upload a vectorizer pickle in the app sidebar at runtime (common filenames: `vectorizer.pkl`, `tfidf.pkl`, `count_vectorizer.pkl`).
	- Or place the vectorizer pickle next to the app with one of the above filenames and the app will try to load it automatically.
- Preferred approach: save a scikit-learn `Pipeline` that includes the vectorizer and estimator (e.g., `Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression())])`) and pickle that pipeline as `spam_mail_model.pkl`.

Contact
- For help or improvements, open an issue in this repository or message the project maintainer.
