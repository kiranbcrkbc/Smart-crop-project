# app.py
from flask import Flask, request, jsonify, render_template_string, send_file
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import os
import csv

app = Flask(__name__)

# --------------------------
# Train a tiny demo ML model
# --------------------------
# We create a small synthetic training set mapping simple rules to crops.
# This is just to demonstrate ML — in a real project you'd use a real dataset.

train_data = [
    # N, P, K, temp, humidity, ph, rainfall, crop
    [80, 40, 40, 25, 85, 5.2, 300, "Rice"],
    [10, 10, 10, 22, 45, 7.0, 60, "Wheat"],
    [20, 20, 20, 32, 55, 6.5, 120, "Maize"],
    [15, 10, 18, 30, 90, 6.0, 400, "Sugarcane"],
    [25, 30, 25, 28, 70, 7.8, 80, "Cotton"],
    [12, 12, 12, 20, 40, 6.8, 50, "Wheat"],
    [75, 35, 50, 26, 82, 5.0, 320, "Rice"],
    [30, 30, 20, 31, 60, 6.4, 150, "Maize"],
    [40, 25, 25, 29, 75, 6.2, 210, "Sugarcane"],
    [18, 15, 15, 33, 50, 6.7, 90, "Maize"]
]

df = pd.DataFrame(train_data, columns=[
    "nitrogen", "phosphorus", "potassium",
    "temperature", "humidity", "ph", "rainfall", "crop"
])
X = df[["nitrogen", "phosphorus", "potassium", "temperature", "humidity", "ph", "rainfall"]]
y = df["crop"]

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# --------------------------
# CSV logging setup
# --------------------------
CSV_FILE = "data.csv"
CSV_HEADERS = ["nitrogen", "phosphorus", "potassium", "temperature", "humidity", "ph", "rainfall", "predicted_crop"]

if not os.path.exists(CSV_FILE):
    # create and write headers
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADERS)

# --------------------------
# HTML page (styled)
# --------------------------
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Smart Crop Recommendation</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root {
      --bg: #f3f7f9;
      --card: #ffffff;
      --accent: #2b9348;
      --muted: #666;
    }
    body { font-family: Arial, sans-serif; background: linear-gradient(180deg,#e6f3ea,#f3f7f9); margin:0; padding:20px; }
    .container { max-width:900px; margin:20px auto; }
    .hero {
      background: url('https://images.unsplash.com/photo-1501004318641-b39e6451bec6?auto=format&fit=crop&w=1400&q=60') center/cover no-repeat;
      border-radius:12px; color: white; padding: 36px; box-shadow: 0 8px 30px rgba(17,24,39,0.12);
    }
    .hero h1 { margin:0 0 8px 0; font-size:28px; text-shadow:0 2px 6px rgba(0,0,0,0.35);}
    .hero p { margin:0 0 0 0; opacity:0.95;}
    .card { background: var(--card); border-radius:10px; padding:18px; margin-top:18px; box-shadow: 0 6px 20px rgba(10,15,30,0.06); }
    label { display:block; margin:8px 0 4px 0; font-weight:600; color:#222; }
    input { width:100%; padding:8px 10px; border-radius:6px; border:1px solid #ddd; box-sizing:border-box; }
    .grid { display:grid; grid-template-columns: repeat(2, 1fr); gap:12px; }
    .full { grid-column: 1 / -1; }
    button { background: var(--accent); color:white; padding:10px 14px; border: none; border-radius:8px; cursor:pointer; font-weight:700; margin-top:10px;}
    .result { margin-top:16px; padding:12px; border-radius:8px; background:#e9fff0; color:#0b6623; font-weight:700; }
    .small { color:var(--muted); font-size:13px; margin-top:6px; }
    .download { background:#1e3a8a; margin-left:8px; }
    @media (max-width:600px) { .grid { grid-template-columns: 1fr; } .hero h1 { font-size:20px; } }
  </style>
</head>
<body>
  <div class="container">
    <div class="hero">
      <h1>🌾 Smart Crop Recommendation System</h1>
      <p>Enter simple farm & soil details below and get a recommended crop (demo ML model).</p>
    </div>

    <div class="card">
      <form method="POST" id="theForm">
        <div class="grid">
          <div>
            <label>Nitrogen (N)</label>
            <input name="nitrogen" type="number" step="any" value="10" required>
          </div>
          <div>
            <label>Phosphorus (P)</label>
            <input name="phosphorus" type="number" step="any" value="10" required>
          </div>
          <div>
            <label>Potassium (K)</label>
            <input name="potassium" type="number" step="any" value="10" required>
          </div>
          <div>
            <label>Temperature (°C)</label>
            <input name="temperature" type="number" step="any" value="25" required>
          </div>
          <div>
            <label>Humidity (%)</label>
            <input name="humidity" type="number" step="any" value="60" required>
          </div>
          <div>
            <label>Soil pH</label>
            <input name="ph" type="number" step="any" value="6.5" required>
          </div>
          <div class="full">
            <label>Rainfall (mm)</label>
            <input name="rainfall" type="number" step="any" value="200" required>
          </div>
        </div>

        <div style="margin-top:12px;">
          <button type="submit">Get Recommendation</button>
          <button id="downloadBtn" type="button" class="download" style="padding:10px 12px;">Download CSV</button>
          <div class="small">Submissions are saved to <code>data.csv</code> in this folder.</div>
        </div>
      </form>

      {% if result %}
        <div class="result">✅ Recommended Crop: {{ result }} </div>
      {% endif %}
    </div>

    <div style="margin-top:12px; text-align:center; color:#444; font-size:13px;">
      Demo ML model: decision tree trained on a tiny sample dataset. Good for demo and submission.
    </div>
  </div>

  <script>
    document.getElementById('downloadBtn').addEventListener('click', () => {
      window.location.href = '/download';
    });

    // Optional: make the form submit without leaving page
    const form = document.getElementById('theForm');
    form.addEventListener('submit', async (e) => {
      // allow normal POST (server-side) by default; if you prefer AJAX, uncomment below
      // e.preventDefault();
      // const data = Object.fromEntries(new FormData(form));
      // const resp = await fetch('/predict-json', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(data)});
      // const json = await resp.json();
      // alert('Recommended: ' + json.predicted_crop);
    });
  </script>
</body>
</html>
"""

# --------------------------
# Routes
# --------------------------

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        try:
            # read values from form
            nitrogen = float(request.form.get("nitrogen", 10))
            phosphorus = float(request.form.get("phosphorus", 10))
            potassium = float(request.form.get("potassium", 10))
            temperature = float(request.form.get("temperature", 25))
            humidity = float(request.form.get("humidity", 60))
            ph_val = float(request.form.get("ph", 6.5))
            rainfall = float(request.form.get("rainfall", 200))
        except Exception:
            result = "Invalid input - please enter numbers"
            return render_template_string(HTML_PAGE, result=result)

        # predict via model
        x = [[nitrogen, phosphorus, potassium, temperature, humidity, ph_val, rainfall]]
        pred = model.predict(x)[0]

        # save to CSV
        with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([nitrogen, phosphorus, potassium, temperature, humidity, ph_val, rainfall, pred])

        result = pred

    return render_template_string(HTML_PAGE, result=result)

@app.route("/predict-json", methods=["POST"])
def predict_json():
    data = request.get_json() or {}
    try:
        nitrogen = float(data.get("nitrogen", 10))
        phosphorus = float(data.get("phosphorus", 10))
        potassium = float(data.get("potassium", 10))
        temperature = float(data.get("temperature", 25))
        humidity = float(data.get("humidity", 60))
        ph_val = float(data.get("ph", 6.5))
        rainfall = float(data.get("rainfall", 200))
    except Exception:
        return jsonify({"error": "Invalid input values"}), 400

    x = [[nitrogen, phosphorus, potassium, temperature, humidity, ph_val, rainfall]]
    pred = model.predict(x)[0]

    # append to CSV
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([nitrogen, phosphorus, potassium, temperature, humidity, ph_val, rainfall, pred])

    return jsonify({"predicted_crop": pred})

@app.route("/download")
def download_csv():
    # allow user to download the saved CSV
    if os.path.exists(CSV_FILE):
        return send_file(CSV_FILE, as_attachment=True)
    else:
        return "No data yet", 404

# --------------------------
# Run server
# --------------------------
if __name__ == "__main__":
    print("Starting upgraded Smart Crop server at http://127.0.0.1:5000")
    app.run(debug=True)
