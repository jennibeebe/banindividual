'''
Our sample flask prediction application

Install these three libraries: 
pip install flask 
pip install flask-restful 
pip install flasgger
'''

import pickle
from flask import Flask, request
from flasgger import Swagger

app = Flask(__name__)
swagger = Swagger(app)

# Load the trained machine learning model that we downloaded
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


# Prediction API
@app.route('/models/predict', methods=['POST'])
def predict():
    """
    Make a prediction based on input features
    ---
    tags:
      - Diabetes Model ML Inference
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            pregnancies:
              type: number
              example: 2
            glucose:
              type: number
              example: 120
            blood_pressure:
              type: number
              example: 70
            skin_thickness:
              type: number
              example: 20
            insulin:
              type: number
              example: 85
            bmi:
              type: number
              example: 25.5
            diabetes_pedigree_function:
              type: number
              example: 0.5
            age:
              type: number
              example: 33
    responses:
      200:
        description: The prediction result
        schema:
          type: object
          properties:
            predictionOutcome:
              type: integer
              example: 1
            inputFeatures:
              type: object
    """
    # Get user input from the JSON payload passed by the user
    data = request.get_json()

    # Extracting features from the JSON input
    user_input = [
        float(data['pregnancies']),
        float(data['glucose']),
        float(data['blood_pressure']),
        float(data['skin_thickness']),
        float(data['insulin']),
        float(data['bmi']),
        float(data['diabetes_pedigree_function']),
        float(data['age'])
    ]

    # Make a prediction using the model
    prediction = model.predict([user_input])

    # Return the result to the user as JSON
    response = {
        "predictionOutcome": int(prediction[0]),
        "inputFeatures": data
    }
    return response

if __name__ == '__main__':
    app.run(debug=True)
