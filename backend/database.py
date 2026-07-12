import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "predictiveengine.db")

def get_db():
    """Get database connection"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database schema"""
    conn = get_db()
    cursor = conn.cursor()

    # Predictions Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            created_at TEXT NOT NULL,
            engine_type TEXT NOT NULL,
            health_score INTEGER NOT NULL,
            risk_level TEXT NOT NULL,
            failure_prob REAL NOT NULL,
            safe_days INTEGER NOT NULL,
            rpm REAL NOT NULL,
            torque REAL NOT NULL,
            wear REAL NOT NULL,
            air_temp REAL NOT NULL,
            process_temp REAL NOT NULL,
            fail_prediction INTEGER NOT NULL
        )
    ''')

    # Migration: Add user_id column if it doesn't exist in existing databases
    try:
        cursor.execute("ALTER TABLE predictions ADD COLUMN user_id INTEGER")
    except sqlite3.OperationalError:
        pass

    # Users Table
    cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fullname TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT NOT NULL,
            address TEXT NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()

def save_prediction(form_data, result, user_id=None):
    """Save prediction to database"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO predictions (
            user_id, created_at, engine_type, health_score, risk_level, 
            failure_prob, safe_days, rpm, torque, wear, 
            air_temp, process_temp, fail_prediction
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_id,
        result['time'],
        result['engine'],
        result['health'],
        result['risk'],
        result['prob'],
        result['days'],
        result['rpm'],
        result['torque'],
        result['wear'],
        float(form_data.get('air_temp', 298.5)),
        float(form_data.get('process_temp', 309)),
        result['fail']
    ))
    
    conn.commit()
    conn.close()

def get_predictions(user_id=None):
    """Get all predictions from database, optionally filtered by user_id"""
    conn = get_db()
    cursor = conn.cursor()
    
    if user_id is not None:
        cursor.execute('SELECT * FROM predictions WHERE user_id = ? ORDER BY id DESC', (user_id,))
    else:
        cursor.execute('SELECT * FROM predictions ORDER BY id DESC')
    rows = cursor.fetchall()
    conn.close()
    
    predictions = []
    for row in rows:
        predictions.append({
            'id': row['id'],
            'created_at': row['created_at'],
            'engine_type': row['engine_type'],
            'health_score': row['health_score'],
            'risk_level': row['risk_level'],
            'failure_prob': row['failure_prob'],
            'safe_days': row['safe_days'],
            'rpm': row['rpm'],
            'torque': row['torque'],
            'wear': row['wear'],
            'air_temp': row['air_temp'],
            'process_temp': row['process_temp'],
            'fail_prediction': row['fail_prediction']
        })
    
    return predictions


def get_stats(user_id=None):
    """Get database statistics, optionally filtered by user_id"""
    conn = get_db()
    cursor = conn.cursor()
    
    if user_id is not None:
        # Total predictions
        cursor.execute('SELECT COUNT(*) as count FROM predictions WHERE user_id = ?', (user_id,))
        total = cursor.fetchone()['count']
        
        # Healthy predictions (safe risk level)
        cursor.execute('SELECT COUNT(*) as count FROM predictions WHERE risk_level = ? AND user_id = ?', ('safe', user_id))
        healthy = cursor.fetchone()['count']
        
        # Critical predictions (danger risk level)
        cursor.execute('SELECT COUNT(*) as count FROM predictions WHERE risk_level = ? AND user_id = ?', ('danger', user_id))
        critical = cursor.fetchone()['count']
    else:
        # Total predictions
        cursor.execute('SELECT COUNT(*) as count FROM predictions')
        total = cursor.fetchone()['count']
        
        # Healthy predictions (safe risk level)
        cursor.execute('SELECT COUNT(*) as count FROM predictions WHERE risk_level = ?', ('safe',))
        healthy = cursor.fetchone()['count']
        
        # Critical predictions (danger risk level)
        cursor.execute('SELECT COUNT(*) as count FROM predictions WHERE risk_level = ?', ('danger',))
        critical = cursor.fetchone()['count']
    
    conn.close()
    
    return {
        'total': total,
        'healthy': healthy,
        'critical': critical
    }
def register_user(fullname, email, phone, address, password):

    conn = get_db()
    cursor = conn.cursor()

    try:

        cursor.execute("""
            INSERT INTO users
            (fullname,email,phone,address,password)
            VALUES (?,?,?,?,?)
        """, (
            fullname,
            email,
            phone,
            address,
            password
        ))

        conn.commit()

        print("USER INSERTED SUCCESSFULLY")

        return True

    except Exception as e:

        print("DATABASE ERROR :", e)

        return False

    finally:

        conn.close()


def get_user_by_email(email):
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM users WHERE email=? OR fullname=?",
        (email, email)
    )

    user = cursor.fetchone()

    conn.close()
    return user

def login_user(email, password):
    """
    Login user using email and password
    """
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM users
        WHERE email=? AND password=?
    """, (email, password))

    user = cursor.fetchone()

    conn.close()

    return user

def update_user_password(email, new_password):
    """
    Update a user's password
    """
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("UPDATE users SET password=? WHERE email=?", (new_password, email))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        print("DATABASE ERROR during password reset:", e)
        return False
    finally:
        conn.close()