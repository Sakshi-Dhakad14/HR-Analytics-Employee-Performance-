import io
import os
import json
import joblib
import pandas as pd
from functools import wraps
from flask import (Flask, render_template, request, redirect,
                   url_for, session, jsonify, send_file)
from werkzeug.security import check_password_hash, generate_password_hash
from utils.preprocess import prepare_features

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")

COMPANY_NAME = "Acme Corp"
USERS = {"admin": generate_password_hash("admin123")}

# ---------- Data aur model ----------
DATA_PATH = "data/IBM-HR-Analytics-Employee-Attrition-and-Performance-Revised.csv"
df = pd.read_csv(DATA_PATH)
df["EmployeeID"] = df.index + 1
model = joblib.load("models/best_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")


def add_risk(data):
    X = prepare_features(data, feature_columns)
    out = data.copy()
    out["RiskProb"] = model.predict_proba(X)[:, 1].round(3)
    out["RiskLevel"] = pd.cut(
        out["RiskProb"], [-0.01, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"]
    ).astype(str)
    return out


df_scored = add_risk(df)

# Bands (Explorer ke "Group by" ke liye)
df_scored["AgeBand"] = pd.cut(df_scored["Age"], [0, 24, 34, 44, 54, 100],
                              labels=["Under 25", "25-34", "35-44", "45-54", "55+"])
df_scored["TenureBand"] = pd.cut(df_scored["YearsAtCompany"], [-1, 2, 5, 10, 100],
                                 labels=["0-2 yrs", "3-5 yrs", "6-10 yrs", "10+ yrs"])
df_scored["IncomeBand"] = pd.cut(df_scored["MonthlyIncome"], [0, 3000, 6000, 10000, 10**9],
                                 labels=["Under $3k", "$3k-6k", "$6k-10k", "$10k+"])

FEATURES = df.drop(columns=["Attrition", "EmployeeID"])
DEFAULTS = FEATURES.mode().iloc[0].to_dict()
NUMERIC_COLS = FEATURES.select_dtypes("number").columns.tolist()
ROLE_MEDIAN = df.groupby("JobRole")["MonthlyIncome"].median().to_dict()

KEY_FIELDS = ["Age", "MonthlyIncome", "OverTime", "JobRole", "Department",
              "BusinessTravel", "MaritalStatus", "DistanceFromHome",
              "TotalWorkingYears", "YearsAtCompany", "YearsSinceLastPromotion",
              "WorkLifeBalance", "JobSatisfaction", "StockOptionLevel"]
KEY_FIELDS = [c for c in KEY_FIELDS if c in FEATURES.columns]

ORDERS = [["Low", "Medium", "High", "Very High"],
          ["Bad", "Good", "Better", "Best"]]


def order_options(opts):
    for o in ORDERS:
        if set(opts) <= set(o):
            return [x for x in o if x in opts]
    return sorted(opts)


@app.context_processor
def inject_globals():
    return dict(company=COMPANY_NAME, current_user=session.get("user"))


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


def filtered_df():
    d = df_scored
    for col in ["Department", "JobRole", "Gender", "OverTime"]:
        val = request.args.get(col)
        if val and val != "All":
            d = d[d[col] == val]
    for key, col in [("age", "Age"), ("inc", "MonthlyIncome")]:
        lo = request.args.get(key + "_min")
        hi = request.args.get(key + "_max")
        if lo:
            d = d[d[col] >= float(lo)]
        if hi:
            d = d[d[col] <= float(hi)]
    return d


def num(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def risk_factors(r):
    """Rule-based risk factors + suggested retention action."""
    f = []
    if str(r.get("OverTime")) == "Yes":
        f.append(("Works overtime", "Review workload and cap overtime hours"))
    if num(r.get("YearsSinceLastPromotion")) >= 3:
        f.append(("3+ years since last promotion",
                  "Schedule a career-growth and promotion review"))
    med = ROLE_MEDIAN.get(r.get("JobRole"))
    if med and num(r.get("MonthlyIncome")) < 0.85 * med:
        f.append(("Income well below role median",
                  "Run a compensation benchmark for this role"))
    if str(r.get("WorkLifeBalance")).lower() in ("bad", "low"):
        f.append(("Poor work-life balance",
                  "Offer flexible hours or remote-work options"))
    if str(r.get("JobSatisfaction")).lower() == "low":
        f.append(("Low job satisfaction",
                  "Hold a 1-on-1 to understand concerns and role fit"))
    if str(r.get("BusinessTravel")) == "Travel_Frequently":
        f.append(("Frequent business travel",
                  "Rebalance travel load across the team"))
    if num(r.get("DistanceFromHome")) >= 20:
        f.append(("Long commute", "Consider hybrid or transport support"))
    if num(r.get("YearsAtCompany"), 99) <= 2:
        f.append(("New joiner (2 years or less)",
                  "Strengthen onboarding and assign a mentor"))
    return [dict(factor=a, action=b) for a, b in f]


# ---------- Auth ----------
@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        u = request.form.get("username", "")
        p = request.form.get("password", "")
        if u in USERS and check_password_hash(USERS[u], p):
            session["user"] = u
            return redirect(url_for("overview"))
        error = "Galat username ya password"
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------- Pages ----------
@app.route("/")
@login_required
def home():
    return redirect(url_for("overview"))


@app.route("/overview")
@login_required
def overview():
    return render_template("overview.html")


@app.route("/explorer")
@login_required
def explorer():
    return render_template("explorer.html")


@app.route("/predict")
@login_required
def predict_page():
    return render_template("predict.html")


@app.route("/high-risk")
@login_required
def high_risk_page():
    return render_template("high_risk.html")


@app.route("/model-performance")
@login_required
def model_performance():
    return render_template("model_performance.html")


# ---------- APIs ----------
@app.route("/api/filters")
@login_required
def api_filters():
    out = {c: sorted(df[c].unique().tolist())
           for c in ["Department", "JobRole", "Gender", "OverTime"]}
    out["ranges"] = {
        "Age": [int(df["Age"].min()), int(df["Age"].max())],
        "MonthlyIncome": [int(df["MonthlyIncome"].min()),
                          int(df["MonthlyIncome"].max())],
    }
    return jsonify(out)


@app.route("/api/alerts")
@login_required
def api_alerts():
    high = int((df_scored["RiskLevel"] == "High").sum())
    return jsonify(high=high, pct=round(high / len(df_scored) * 100, 1))


@app.route("/api/kpis")
@login_required
def api_kpis():
    d = filtered_df()
    if len(d) == 0:
        return jsonify(total=0, attrition_rate=0, avg_income=0, avg_tenure=0,
                       high_risk=0, attrition_cost=0)
    left = int((d["Attrition"] == "Yes").sum())
    avg_income = d["MonthlyIncome"].mean()
    return jsonify(
        total=len(d),
        attrition_rate=round(left / len(d) * 100, 1),
        avg_income=int(avg_income),
        avg_tenure=round(d["YearsAtCompany"].mean(), 1),
        high_risk=int((d["RiskLevel"] == "High").sum()),
        # Assumption: replacing an employee costs ~50% of annual salary
        attrition_cost=int(left * avg_income * 12 * 0.5),
    )


ALLOWED_GROUPS = ["Department", "JobRole", "OverTime", "MaritalStatus",
                  "WorkLifeBalance", "JobSatisfaction", "BusinessTravel",
                  "Gender", "EducationField", "AgeBand", "TenureBand",
                  "IncomeBand"]


@app.route("/api/attrition-by/<col>")
@login_required
def api_attrition_by(col):
    if col not in ALLOWED_GROUPS:
        return jsonify(error="invalid column"), 400
    d = filtered_df()
    g = d.groupby(col, observed=True).agg(
        total=("Attrition", "size"),
        left=("Attrition", lambda s: (s == "Yes").sum()),
    )
    g["rate"] = (g["left"] / g["total"] * 100).round(1)
    return jsonify(labels=[str(x) for x in g.index.tolist()],
                   rate=g["rate"].tolist(),
                   left=g["left"].astype(int).tolist(),
                   total=g["total"].astype(int).tolist())


@app.route("/api/risk-distribution")
@login_required
def api_risk_distribution():
    vc = filtered_df()["RiskLevel"].value_counts()
    labels = ["Low", "Medium", "High"]
    return jsonify(labels=labels, values=[int(vc.get(k, 0)) for k in labels])


@app.route("/api/heatmap")
@login_required
def api_heatmap():
    d = filtered_df().copy()
    d["left"] = (d["Attrition"] == "Yes").astype(int)
    piv = d.pivot_table(index="JobRole", columns="Department",
                        values="left", aggfunc="mean") * 100
    values = [[None if pd.isna(v) else round(float(v), 1) for v in row]
              for row in piv.values]
    return jsonify(rows=piv.index.tolist(), cols=piv.columns.tolist(),
                   values=values)


@app.route("/api/insights")
@login_required
def api_insights():
    d = filtered_df()
    if len(d) < 30:
        return jsonify([])
    out = []
    is_left = d["Attrition"] == "Yes"

    ot = d[d["OverTime"] == "Yes"]
    no_ot = d[d["OverTime"] == "No"]
    if len(ot) and len(no_ot):
        a = (ot["Attrition"] == "Yes").mean() * 100
        b = (no_ot["Attrition"] == "Yes").mean() * 100
        if b > 0:
            out.append(f"Employees working overtime leave <b>{a / b:.1f}x</b> "
                       f"more often ({a:.0f}% vs {b:.0f}%).")

    role = d.groupby("JobRole")["Attrition"].apply(
        lambda s: (s == "Yes").mean() * 100).sort_values(ascending=False)
    out.append(f"Highest-attrition role: <b>{role.index[0]}</b> "
               f"({role.iloc[0]:.0f}%).")

    if is_left.any() and (~is_left).any():
        gap = d[~is_left]["MonthlyIncome"].mean() - d[is_left]["MonthlyIncome"].mean()
        if gap > 0:
            out.append(f"Employees who left earned <b>${gap:,.0f}</b> less per "
                       f"month on average than those who stayed.")

    new = d[d["YearsAtCompany"] <= 2]
    if len(new):
        rate = (new["Attrition"] == "Yes").mean() * 100
        out.append(f"Employees with 2 years or less at the company have "
                   f"<b>{rate:.0f}%</b> attrition.")

    high = int((d["RiskLevel"] == "High").sum())
    out.append(f"<b>{high}</b> employees are currently flagged as High risk.")
    return jsonify(out)


@app.route("/api/form-schema")
@login_required
def api_form_schema():
    out = []
    for c in KEY_FIELDS:
        s = FEATURES[c]
        if pd.api.types.is_numeric_dtype(s):
            out.append(dict(name=c, type="number", min=float(s.min()),
                            max=float(s.max()), default=float(s.median())))
        else:
            out.append(dict(name=c, type="select",
                            options=order_options(s.unique().tolist()),
                            default=str(DEFAULTS[c])))
    return jsonify(out)


@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    data = dict(DEFAULTS)
    data.update(request.get_json())
    row = pd.DataFrame([data])
    for c in NUMERIC_COLS:
        row[c] = pd.to_numeric(row[c])
    prob = float(model.predict_proba(prepare_features(row, feature_columns))[0, 1])
    level = "High" if prob > 0.6 else "Medium" if prob > 0.3 else "Low"
    return jsonify(probability=round(prob * 100, 1), level=level,
                   factors=risk_factors(data))


@app.route("/api/employee/<int:emp_id>")
@login_required
def api_employee(emp_id):
    row = df_scored[df_scored["EmployeeID"] == emp_id]
    if row.empty:
        return jsonify(error="not found"), 404
    r = row.iloc[0]
    details = {}
    for c in KEY_FIELDS:
        v = r[c]
        details[c] = v.item() if hasattr(v, "item") else v
    return jsonify(id=emp_id, probability=round(float(r["RiskProb"]) * 100, 1),
                   level=r["RiskLevel"], details=details,
                   factors=risk_factors(r))


@app.route("/api/high-risk")
@login_required
def api_high_risk():
    top = int(request.args.get("top", 20))
    cols = ["EmployeeID", "Age", "Department", "JobRole", "MonthlyIncome",
            "OverTime", "YearsAtCompany", "RiskProb", "RiskLevel"]
    d = filtered_df().sort_values("RiskProb", ascending=False).head(top)
    return jsonify(d[cols].to_dict(orient="records"))


@app.route("/download/high-risk")
@login_required
def download_high_risk():
    d = filtered_df()
    d = d[d["RiskLevel"] == "High"].sort_values("RiskProb", ascending=False)
    buf = io.BytesIO(d.to_csv(index=False).encode("utf-8"))
    return send_file(buf, mimetype="text/csv", as_attachment=True,
                     download_name="high_risk_employees.csv")


@app.route("/api/model-metrics")
@login_required
def api_model_metrics():
    with open("models/metrics.json") as f:
        return jsonify(json.load(f))


if __name__ == "__main__":
    app.run(debug=True)