# 📩 Spam Message Classifier

A machine-learning project that classifies SMS messages as **spam** or **ham** (not spam), with a Streamlit web demo.

## Features
- TF-IDF text features (unigrams + bigrams)
- Compares **Naive Bayes** vs **Logistic Regression**, saves the better one
- Evaluation: accuracy, precision/recall/F1, confusion matrix
- Interactive Streamlit app

## Project structure
```
spam-classifier/
├── app.py            # Streamlit demo
├── train.py          # training + evaluation
├── requirements.txt
├── data/sample_spam.csv   # tiny fallback dataset
├── models/           # saved model (created by train.py)
└── images/           # confusion matrix (created by train.py)
```

## Run it
```bash
python -m venv venv
venv\Scripts\activate        # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python train.py              # downloads the full SMS Spam dataset automatically
streamlit run app.py
```

## Results
Tested on the full SMS Spam dataset (25% held-out test set):

| Model | Accuracy | Spam F1 |
|---|---|---|
| Naive Bayes | 96.0% | 0.83 |
| Logistic Regression | 97.9% | 0.92 |

![Confusion matrix](images/confusion_matrix.png)

## What I learned
- Text preprocessing and TF-IDF
- Why F1 matters more than accuracy on imbalanced data
- Saving models with joblib and serving them with Streamlit

## Dataset
[SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (UCI). Falls back to a tiny sample if offline.
"# spam-message-calculator" 
