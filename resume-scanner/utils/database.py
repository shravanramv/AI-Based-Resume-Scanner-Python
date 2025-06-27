import sqlite3
import os

# Absolute path to the shared database file in `databases/`
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "databases", "users.db"))

def connect_db():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = connect_db()
    c = conn.cursor()

    # Unified users table for both roles
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            role TEXT CHECK(role IN ('recruiter', 'applicant')),
            company TEXT
        )
    ''')

    # Jobs table
    c.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recruiter_username TEXT,
            job_title TEXT,
            job_description TEXT
        )
    ''')

    # Applications table
    c.execute('''
        CREATE TABLE IF NOT EXISTS applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            applicant_username TEXT,
            job_id INTEGER,
            resume BLOB,
            resume_filename TEXT,
            FOREIGN KEY (job_id) REFERENCES jobs(id)
        )
    ''')

    conn.commit()
    conn.close()

# --- Auth Functions (for login.py) ---
def get_user_role(username, password):
    conn = connect_db()
    c = conn.cursor()
    c.execute("SELECT role FROM users WHERE username = ? AND password = ?", (username, password))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

def register_user(username, password, role, company):
    conn = connect_db()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password, role, company) VALUES (?, ?, ?, ?)",
                  (username, password, role, company))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()
