from flask import Flask, request, jsonify
from app.prediction import Model

import json

app = Flask(__name__)

model = Model()

@app.route('/predict', methods=['POST']) # name is whatever baad mein
def predict():
    
    try:
        image = request.json
        # print(image)

        result = model.run_model(image)

        result2 = {
            "array": result[0].tolist()
        }

        result3 = json.dumps(result2, indent=4)
        return result3

    except Exception as exception:
        print(exception)
        return jsonify({"error": "could not predict"})

@app.route('/')
def home():
    return 'home page'


    