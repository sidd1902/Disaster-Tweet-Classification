from flask import Flask, render_template, request
import pickle
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load the model "C:\Users\skhai\project 7\notebooks\regression_model.pkl"
model_path = r'C:\Users\skhai\project 7\notebooks\regression_model.pkl'
if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
    except Exception as e:
        model = None
        print(f"Error loading model: {e}")
else:
    model = None
    print(f"Model file not found: {model_path}")

# Prediction function
def model_predict(text_data):
    if model:
        try:
            # Assuming 'clean_text' function is defined to preprocess text if necessary
            # cleaned_text = clean_text(text_data)
            preds = model.predict([text_data])
            pred = (preds[0] > 0.42).astype(np.int32)  # Adjust as per the model’s output
            return pred
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Error in prediction"
    else:
        return "Model not loaded"

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        tweet = request.form['cleaned_text']
        prediction = model_predict(tweet)
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
