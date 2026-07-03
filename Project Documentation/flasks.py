from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# LOAD MODEL
with open("flood_model.pkl", "rb") as f:
    artifacts = pickle.load(f)

model = artifacts["model"]
model_columns = artifacts["columns"]

# HOME PAGE
@app.route('/')
def home():
    return render_template("index.html")

# PREDICTION API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        df = pd.DataFrame([data])

        # convert safely
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

        # FEATURE ENGINEERING
        df["Climate_Risk"] = df["MonsoonIntensity"] + df["ClimateChange"] + df["CoastalVulnerability"]

        df["Infrastructure_Risk"] = df["DeterioratingInfrastructure"] + df["DrainageSystems"] + df["DamsQuality"]

        df["Human_Impact"] = df["Urbanization"] + df["Deforestation"] + df["Encroachments"] + df["PopulationScore"]

        df["Environmental_Risk"] = df["WetlandLoss"] + df["Siltation"] + df["AgriculturalPractices"]

        df["Disaster_Management"] = df["IneffectiveDisasterPreparedness"] + df["InadequatePlanning"] + df["PoliticalFactors"]

        # ALIGN COLUMNS
        df = df.reindex(columns=model_columns, fill_value=0)

        # PREDICT
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        return jsonify({
            "Flood_Prediction": int(pred),
            "Flood_Probability": float(prob)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# RUN SERVER
if __name__ == '__main__':
    app.run(debug=False)