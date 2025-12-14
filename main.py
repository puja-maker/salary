from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("salary_model.pkl")


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            job_experience_years = float(request.form["job_experience_years"])
            previous_salary = float(request.form["previous_salary"])
            study_hours_per_day = float(request.form["study_hours_per_day"])
            python_skill = float(request.form["python_skill"])
            ml_skill = float(request.form["ml_skill"])

            # Prepare input array
            test_array = np.array([[  
                job_experience_years,
                previous_salary,
                study_hours_per_day,
                python_skill,
                ml_skill
            ]])

            prediction = model.predict(test_array)[0]

        except Exception as e:
            error = "Invalid input. Please enter valid values."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error
    )


# -------- API Endpoint (optional) --------
@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json()

    try:
        test_array = np.array([[  
            float(data["job_experience_years"]),
            float(data["previous_salary"]),
            float(data["study_hours_per_day"]),
            float(data["python_skill"]),
            float(data["ml_skill"])
        ]])

        prediction = model.predict(test_array)[0]

        return jsonify({"predicted_salary": prediction})

    except Exception as e:
        return jsonify({"error": "Invalid input"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
