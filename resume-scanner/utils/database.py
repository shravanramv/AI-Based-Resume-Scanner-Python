import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "users.db")

def connect_db():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = connect_db()
    c = conn.cursor()

    # Recruiters table
    c.execute('''
        CREATE TABLE IF NOT EXISTS recruiters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    ''')

    # Applicants table
    c.execute('''
        CREATE TABLE IF NOT EXISTS applicants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
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

# Recruiter functions
def create_recruiter(username, password):
    conn = connect_db()
    c = conn.cursor()
    try:
        c.execute('INSERT INTO recruiters (username, password) VALUES (?, ?)', (username, password))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    conn.close()

def authenticate_recruiter(username, password):
    conn = connect_db()
    c = conn.cursor()
    c.execute('SELECT * FROM recruiters WHERE username = ? AND password = ?', (username, password))
    user = c.fetchone()
    conn.close()
    return user

# Applicant functions
def create_applicant(username, password):
    conn = connect_db()
    c = conn.cursor()
    try:
        c.execute('INSERT INTO applicants (username, password) VALUES (?, ?)', (username, password))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    conn.close()

def authenticate_applicant(username, password):
    conn = connect_db()
    c = conn.cursor()
    c.execute('SELECT * FROM applicants WHERE username = ? AND password = ?', (username, password))
    user = c.fetchone()
    conn.close()
    return user
