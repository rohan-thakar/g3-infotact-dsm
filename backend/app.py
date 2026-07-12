import os
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from dotenv import load_dotenv
load_dotenv()
 
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
    login_user,
    update_user_password
)
 
BASE = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE, "..", "frontend", "templates"),
    static_folder=os.path.join(BASE, "..", "frontend", "static")
)
app.secret_key = os.environ.get(
    "SECRET_KEY",
    "predictive-maintenance-secret"
)
app.config["SESSION_PERMANENT"] = False

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
    step = request.args.get("step", "login")
 
    return render_template(
        "login.html",
        messages=messages,
        active_tab="login",
        step=step
    )


# ==========================
# Signup Page
# ==========================

@app.route("/signup")
def signup_page():

    if "user_id" in session:
        return redirect("/")

    messages = get_flashed_messages(with_categories=True)
    step = request.args.get("step", "register")

    return render_template(
        "login.html",
        messages=messages,
        active_tab="register",
        step=step
    )


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


# ==========================
# Forgot & Reset Password
# ==========================

def send_recovery_email(recipient_email, code):
    sender_email = os.environ.get("SENDER_EMAIL")
    sender_password = os.environ.get("SENDER_PASSWORD")
    
    if not sender_email or not sender_password:
        print("SMTP Credentials not configured in environment variables.")
        return False
        
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = "Predictive Engine - Password Recovery Code"
        
        body = f"""Hi,

You requested a password reset for your Predictive Engine account.

Your verification code is: {code}

If you did not request this, please ignore this email.

Best regards,
Predictive Engine Team"""
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Using Gmail SMTP Server
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print("SMTP Error occurred during password recovery:", e)
        return False

@app.route("/forgot-password", methods=["POST"])
def forgot_password():
    email = request.form.get("email")
    user = get_user_by_email(email)
    
    if user:
        import random
        code = str(random.randint(100000, 999999))
        session["reset_code"] = code
        session["reset_email"] = email
        
        email_sent = send_recovery_email(email, code)
        if email_sent:
            flash("Verification code has been sent to your email address.", "success")
        else:
            flash(f"Failed to send email. Verification code displayed for development: {code}", "warning")
            
        return redirect(url_for("login_page", step="verify"))
    else:
        flash("Email address not found.", "danger")
        return redirect(url_for("login_page", step="forgot"))


@app.route("/reset-password", methods=["POST"])
def reset_password():
    code = request.form.get("code")
    password = request.form.get("password")
    confirm = request.form.get("confirm_password", password)
    
    if "reset_code" not in session or "reset_email" not in session:
        flash("Session expired. Please try again.", "danger")
        return redirect(url_for("login_page"))
        
    if code != session["reset_code"]:
        flash("Invalid verification code.", "danger")
        return redirect(url_for("login_page", step="verify"))
        
    if password != confirm:
        flash("Passwords do not match.", "danger")
        return redirect(url_for("login_page", step="verify"))
        
    email = session["reset_email"]
    if update_user_password(email, password):
        session.pop("reset_code", None)
        session.pop("reset_email", None)
        flash("Password reset successful. Please sign in.", "success")
        return redirect(url_for("login_page"))
    else:
        flash("Failed to update password. Please try again.", "danger")
        return redirect(url_for("login_page"))

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
    stats = get_stats(session.get("user_id"))

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
            result,
            session["user_id"]
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

    if "user_id" not in session:
        flash("Please sign in to view history.")
        return redirect(url_for("login_page"))

    records = get_predictions(session["user_id"])
    stats = get_stats(session["user_id"])

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
