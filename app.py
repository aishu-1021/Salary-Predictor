from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# ── Load all models & encoders ───────────────────────────────
model_low  = joblib.load('model/model_low.pkl')
model_mid  = joblib.load('model/model_mid.pkl')
model_high = joblib.load('model/model_high.pkl')
encoders   = joblib.load('model/encoders.pkl')
features   = joblib.load('model/features.pkl')

# ── Home route → serves the HTML page ───────────────────────
@app.route('/')
def home():
    # Send unique values to populate dropdowns
    return render_template('index.html',
        experience_levels = ['EN', 'MI', 'SE', 'EX'],
        employment_types  = ['FT', 'PT', 'CT', 'FL'],
        company_sizes     = ['S', 'M', 'L'],
        job_titles        = sorted(encoders['job_title'].classes_.tolist()),
        work_years        = [2020, 2021, 2022, 2023, 2024, 2025]
    )

# ── Predict route → receives form data, returns prediction ───
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # ── Encode inputs using saved encoders ───────────────
        job_title_enc  = encoders['job_title'].transform(
                            [data['job_title']])[0]
        exp_enc        = encoders['experience_level'].transform(
                            [data['experience_level']])[0]
        emp_enc        = encoders['employment_type'].transform(
                            [data['employment_type']])[0]
        size_enc       = encoders['company_size'].transform(
                            [data['company_size']])[0]

        # ── Build input array in correct feature order ───────
        input_data = np.array([[
            int(data['work_year']),
            exp_enc,
            emp_enc,
            job_title_enc,
            int(data['remote_ratio']),
            size_enc
        ]])

        # ── Predict low, mid, high ───────────────────────────
        low  = round(float(model_low.predict(input_data)[0]),  -3)
        mid  = round(float(model_mid.predict(input_data)[0]),  -3)
        high = round(float(model_high.predict(input_data)[0]), -3)

        return jsonify({
            'success': True,
            'low':  f"${low:,.0f}",
            'mid':  f"${mid:,.0f}",
            'high': f"${high:,.0f}"
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)