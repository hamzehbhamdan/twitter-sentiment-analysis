from flask import Flask, render_template, request, jsonify
import stock_analysis as ysa
import pandas as pd

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    
    companies = request.form.getlist("companies")
    start_date = request.form.get("start_date")
    end_date = request.form.get("end_date")
    model_type = request.form.get("model")
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    date_range = pd.date_range("2015-01-01", "2019-12-31")
    
    if start_date < date_range.min() or start_date > date_range.max():
        return render_template("error.html", error_message="Error: Start date must be between Jan 1 2015 and Dec 31 2019")

    if end_date < start_date or end_date > date_range.max():
        return render_template("error.html", error_message="Error: End date must be between start date and Dec 31 2019")

    if start_date == end_date:
        return render_template("error.html", error_message="Error: Start date and end date cannot be the same")

    analysis_results = ysa.analyze_stocks(companies, start_date, end_date, model_type)
    return render_template("results.html", results=analysis_results)

if __name__ == "__main__":
    app.run(debug=True)
