from flask import Flask, request
from flask_restx import Api, Resource
import pickle
import numpy as np

MODELS = {
    "v1": pickle.load(open("models/model_v1.pkl", "rb")),
    "v2": pickle.load(open("models/model_v2.pkl", "rb"))
}

app = Flask(__name__)
api = Api(app, version="2.0", title="Salary Predictor API",
          description="Predict salary based on years of experience using multiple ML models")
ns = api.namespace("predict", description="Salary prediction operations")

@ns.route("/")
class SalaryPredict(Resource):
    @api.doc(params={
        "years_experience": "Years of experience (float)",
        "version": "Model version (v1=Linear Regression, v2=Random Forest)"
    })
    def get(self):
        return self.predict(request.args)

    def post(self):
        if not request.is_json:
            return {"error": "Request must be JSON"}, 400
        return self.predict(request.json)

    def predict(self, data):
        try:
            version = data.get("version", "v1")
            if version not in MODELS:
                return {"error": f"Invalid version. Use one of {list(MODELS.keys())}"}, 400
            years_exp = float(data.get("years_experience"))
            model = MODELS[version]
            prediction = model.predict(np.array([[years_exp]]))[0]
            return {
                "input_years_experience": years_exp,
                "predicted_salary": round(prediction, 2),
                "model_version": version
            }
        except Exception as e:
            return {"error": str(e)}, 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")