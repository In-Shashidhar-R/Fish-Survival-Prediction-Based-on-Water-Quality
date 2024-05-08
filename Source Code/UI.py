import gradio as gr
import numpy as np
from sklearn.ensemble import RandomForestClassifier  # Import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import pandas as pd

def run_fish_survival_predictor():
    # Load data from CSV file
    df = pd.read_csv("C:\\Users\\INFINIX\\Desktop\\Study\\Semester - 4\\22AIE214 - ROBOTICS\\Endsem - Project\\Codes\\realfishdataset.csv")

    # Encode categorical target variable into integers
    label_encoder = LabelEncoder()
    df['fish_encoded'] = label_encoder.fit_transform(df['fish'])

    # Splitting the data into features and target
    X = df[['ph', 'temperature', 'turbidity']]
    y = df['fish_encoded']

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Handling class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_scaled, y)

    # Train Random Forest Classifier
    rf_classifier = RandomForestClassifier(random_state=42)  # Instantiate RandomForestClassifier
    rf_classifier.fit(X_balanced, y_balanced)  # Train the classifier

    # Function to predict fish based on input values using the Random Forest Classifier
    def predict_fish_rf(ph, temperature, turbidity):
        # Scale input features
        input_features = scaler.transform([[ph, temperature, turbidity]])
        # Predict the fish using the Random Forest Classifier
        prediction = rf_classifier.predict(input_features)
        # Inverse transform the prediction to get original labels
        predicted_label = label_encoder.inverse_transform(prediction)
        # Determine image path based on predicted label
        image_path = "C:\\Users\\INFINIX\\Desktop\\Study\\Semester - 4\\22AIE214 - ROBOTICS\\Endsem - Project\\Codes\\UI\\Fish\\" + predicted_label[0] + ".jpg"
        return predicted_label[0], image_path

    # Gradio Interface
    demo = gr.Interface(
        fn=predict_fish_rf,
        inputs=["text", "slider", "text"],
        outputs=["text", "image"],  # Adding image output
        title="Fish Survival Predictor (Random Forest)",
        description="Predicts which fish can survive based on pH, temperature, and turbidity using Random Forest algorithm."
    )
    demo.launch(share=True)

# Run the fish survival predictor
run_fish_survival_predictor()
