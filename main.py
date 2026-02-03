from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Recruitment ML Predictor")

# Allow JS to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load your trained model
model = joblib.load("model.pkl")

# Input schema
class InputData(BaseModel):
    features: list[float]

# Serve HTML page
@app.get("/", response_class=HTMLResponse)
def home():
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>🚀 Recruitment ML Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Roboto', sans-serif;
            margin: 0;
            padding: 0;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            background: linear-gradient(135deg, #6a11cb, #2575fc);
            color: #fff;
        }

        .container {
            background: rgba(0,0,0,0.75);
            padding: 40px;
            border-radius: 20px;
            width: 400px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
            text-align: center;
        }

        h1 {
            margin-bottom: 30px;
            font-size: 28px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #ffeb3b;
        }

        input {
            width: 80%;
            padding: 12px;
            margin: 8px 0;
            border-radius: 10px;
            border: none;
            outline: none;
            font-size: 16px;
        }

        button {
            margin-top: 15px;
            padding: 12px 25px;
            font-size: 16px;
            font-weight: bold;
            color: #fff;
            background: #ff6a00;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        button:hover {
            background: #ff3d00;
        }

        #result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 10px;
            font-size: 18px;
            font-weight: bold;
        }

        input::placeholder {
            color: #g;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Welecome To Recruitment ML Predictor</h1>

        <input id="f1" placeholder="Skills Match Score"><br>
        <input id="f2" placeholder="Project Count"><br>
        <input id="f3" placeholder="Resume Length"><br>
        <input id="f4" placeholder="Github Activity"><br>
        <input id="f5" placeholder="High School (0/1)"><br>
        <input id="f6" placeholder="Masters (0/1)"><br>
        <input id="f7" placeholder="PhD (0/1)"><br>
        <input id="f8" placeholder="Entry (0/1)"><br>
        <input id="f9" placeholder="Mid (0/1)"><br>
        <input id="f10" placeholder="High (0/1)"><br>

        <button onclick="predict()">Predict</button>

        <div id="result"></div>
    </div>

    <script>
    async function predict() {
        let features = [
            parseFloat(f1.value),
            parseFloat(f2.value),
            parseFloat(f3.value),
            parseFloat(f4.value),
            parseFloat(f5.value),
            parseFloat(f6.value),
            parseFloat(f7.value),
            parseFloat(f8.value),
            parseFloat(f9.value),
            parseFloat(f10.value)
        ];

        try {
            let response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({features: features})
            });
            let data = await response.json();
            const resultDiv = document.getElementById("result");
            if (data.result === "Shortlisted") {
                resultDiv.style.background = "rgba(0,255,0,0.3)";
                resultDiv.style.color = "#0a0";
            } else {
                resultDiv.style.background = "rgba(255,0,0,0.3)";
                resultDiv.style.color = "#a00";
            }
            resultDiv.innerHTML = "Result: <b>" + data.result + "</b><br>Confidence: <b>" + data.confidence + "%</b>";
        } catch(err) {
            document.getElementById("result").innerHTML = "Error: " + err;
        }
    }
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    arr = np.array(data.features).reshape(1, -1)
    pred = model.predict(arr)[0]
    confidence = model.predict_proba(arr)[0][1]
    result = "Shortlisted" if pred == 1 else "Rejected"
    return {"result": result, "confidence": round(float(confidence)*100,2)}
