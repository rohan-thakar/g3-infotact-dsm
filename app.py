import os
from datetime import datetime
 
import joblib
import pandas as pd
 
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    get_flashed_messages
)
 
from database import (
    get_db,
    init_db,
    save_prediction,
    get_predictions,
    get_stats,
    register_user,
    get_user_by_email,
    login_user
)
 
app = Flask(__name__)
app.secret_key = os.environ.get(
    "SECRET_KEY",
    "predictive-maintenance-secret"
)
app.config["SESSION_PERMANENT"] = False
 
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL = joblib.load(
    os.path.join(BASE, "predictive_maintenance_model.pkl")
)
 
TYPE_MAP = {"L": 0, "M": 1, "H": 2}
TYPE_LABELS = {"L": "Light", "M": "Medium", "H": "Heavy"}
 
init_db()
 
 
# ==========================
# Prediction Logic
# ==========================
 
def predict_engine(form):
 
    air = float(form["air_temp"])
    proc = float(form["process_temp"])
    rpm = float(form["rpm"])
    torque = float(form["torque"])
    wear = float(form["tool_wear"])
 
    row = {
        "Type": TYPE_MAP.get(form.get("engine_type", "M"), 1),
        "Air temperature [K]": air,
        "Process temperature [K]": proc,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": wear,
        "Temp_Difference": proc - air,
        "Power_Index": rpm * torque,
        "Torque_RPM_Ratio": torque / rpm,
    }
 
    df = pd.DataFrame([row])[list(MODEL.feature_names_in_)]
 
    prob = float(MODEL.predict_proba(df)[0][1])
    fail = int(MODEL.predict(df)[0])
 
    if prob >= 0.6:
 
        risk = "danger"
        msg = "Immediate maintenance required."
 
        days = max(3, int(15 * (1 - prob)))
        health = max(10, int(100 - prob * 100))
 
    elif prob >= 0.3:
 
        risk = "warning"
        msg = "Schedule maintenance soon."
 
        days = max(15, int(40 * (1 - prob)))
        health = max(40, int(75 - prob * 50))
 
    else:
 
        risk = "safe"
        msg = "Engine is healthy."
 
        days = max(60, int(120 * (1 - prob)))
        health = max(70, int(95 - prob * 30))
 
    return {
        "fail": fail,
        "prob": round(prob * 100, 1),
        "risk": risk,
        "msg": msg,
        "days": days,
        "health": health,
        "time": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "rpm": rpm,
        "torque": torque,
        "wear": wear,
        "engine": TYPE_LABELS.get(
            form.get("engine_type", "M"),
            "Medium"
        ),
    }
 
# ==========================
# Login Page
# ==========================
 
@app.route("/login")
def login_page():
 
    if "user_id" in session:
        return redirect("/")
 
    messages = get_flashed_messages(with_categories=True)
 
    return render_template("login.html", messages=messages, active_tab="login")


# ==========================
# Signup Page
# ==========================

@app.route("/signup")
def signup_page():

    if "user_id" in session:
        return redirect("/")

    messages = get_flashed_messages(with_categories=True)

    return render_template("login.html", messages=messages, active_tab="register")


# ==========================
# Register
# ==========================
 
@app.route("/register", methods=["POST"])
def register():

    print("========== REGISTER REQUEST ==========")
    print("FORM DATA:", request.form)

    fullname = request.form.get("fullname")
    email = request.form.get("email")
    phone = request.form.get("phone", "")
    address = request.form.get("address", "")
    password = request.form.get("password")
    confirm = request.form.get("confirm_password", password)

    print("Full Name :", fullname)
    print("Email     :", email)
    print("Phone     :", phone)
    print("Address   :", address)

    if password != confirm:
        flash("Passwords do not match")
        return redirect("/signup")

    existing = get_user_by_email(email)

    if existing:
        flash("Email already registered")
        return redirect("/signup")

    success = register_user(
        fullname,
        email,
        phone,
        address,
        password
    )

    print("REGISTER RESULT:", success)

    if not success:
        flash("Registration failed")
        return redirect("/signup")

    flash("Registration successful")
    return redirect("/login")
 
 
# ==========================
# Login
# ==========================
 
@app.route("/login", methods=["POST"])
def login():

    email = request.form["email"]
    password = request.form["password"]

    user = get_user_by_email(email)

    print("USER FOUND:", user)

    if user and user["password"] == password:

        session.clear()               # Purani session delete
        session.permanent = False     # Browser session only

        session["user_id"] = user["id"]
        session["user_name"] = user["fullname"]

        return redirect("/")

    print("LOGIN FAILED")

    flash("Invalid Email or Password")
    return redirect("/login")
 
 
# ==========================
# Logout
# ==========================
 
@app.route("/logout")
def logout():

    session.clear()

    flash("Logged out successfully.")

    return redirect("/login")

@app.route("/users")
def users():

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT id, fullname, email FROM users")

    rows = cursor.fetchall()

    conn.close()

    return str([dict(x) for x in rows])
 
 
# ==========================
# Home
# ==========================
 
@app.route("/")
def home():

    print("SESSION:", dict(session))

    logged_in = "user_id" in session
    stats = get_stats()

    return render_template(
        "index.html",
        stats=stats,
        username=session.get("user_name"),
        logged_in=logged_in
    )
 
 
# ==========================
# Prediction
# ==========================
 
@app.route("/predict", methods=["POST"])
def predict():
 
    if "user_id" not in session:
        flash("Please sign in to check the engine's health.")
        return redirect(url_for("login_page"))

    try:
 
        result = predict_engine(request.form)
 
        save_prediction(
            request.form,
            result
        )
 
        return render_template(
            "result.html",
            r=result
        )
 
    except Exception as e:
 
        flash(f"Error: {e}")
        return redirect(url_for("home"))
 
 
# ==========================
# History
# ==========================
 
@app.route("/history")
def history():

    # if "user_id" not in session:
    #     return redirect("/login")

    records = get_predictions()
    stats = get_stats()

    return render_template(
        "history.html",
        records=records,
        stats=stats
    )
 
 
# ==========================
# Other Pages
# ==========================
 
@app.route("/about")
def about():
    return render_template("about.html")
 
 
@app.route("/project")
def project():
    return render_template("project.html")
 
 
@app.route("/contact")
def contact():
    return render_template("contact.html")
 
 
# ==========================
# 404
# ==========================
 
@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404
 
 
# if __name__ == "__main__":
 
#     print("\n>>> Server: http://127.0.0.1:5000")
#     print(">>> Database: data/predictiveengine.db\n")
 
#     app.run(
#         host="127.0.0.1",
#         port=5000,
#         debug=False
#     )
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
