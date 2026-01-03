# Credit Card Fraud Detection System

A Streamlit web application that uses Machine Learning (Random Forest) to detect fraudulent credit card transactions.

## Project Structure
- `app.py`: Main Streamlit application
- `train_model.py`: Script to train the ML model
- `requirements.txt`: Project dependencies
- `fraud_detection_model.pkl`: Trained Random Forest model
- `scaler_*.pkl`: Scalers for data preprocessing

## Local Setup

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the App**
    ```bash
    streamlit run app.py
    ```

## Deployment Guide (Streamlit Community Cloud)

This app is ready to be deployed on Streamlit Community Cloud for free.

### Steps:

1.  **Push to GitHub**
    - Create a new repository on GitHub.
    - Push all project files to the repository.
    *(Note: If `creditcard.csv` is larger than 100MB, you might need to use Git LFS or exclude it if it's not needed for inference. For this app, only `.pkl` files are needed for the app to run).*

2.  **Deploy on Streamlit Cloud**
    - Go to [share.streamlit.io](https://share.streamlit.io/).
    - Sign in with GitHub.
    - Click **"New app"**.
    - Select your repository, branch (usually `main`), and main file path (`app.py`).
    - Click **"Deploy"**.

3.  **Done!**
    - Your app will be live in a few minutes.

## Model Info
- **Algorithm**: Random Forest Classifier
- **Features**: Time, Amount, V1-V28 (PCA transformed features)
- **Performance**: High accuracy with balanced class handling.
