from pathlib import Path
import sqlite3
import pandas as pd

DB_PATH = Path("data/gradelens.db")

def load_answers(path):
    df = pd.read_csv(path)
    missing = {"answer_text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df.dropna(subset=["answer_text", "label"]).assign(
        answer_text=lambda x: x.answer_text.astype(str),
        label=lambda x: x.label.astype(int))

def initialize_db(db_path=DB_PATH):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS grades (id INTEGER PRIMARY KEY, student_id TEXT, question_id TEXT, answer_text TEXT NOT NULL, predicted_score INTEGER NOT NULL, teacher_score INTEGER, confidence REAL, created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
        conn.execute("CREATE TABLE IF NOT EXISTS experiments (id INTEGER PRIMARY KEY, name TEXT NOT NULL, accuracy REAL, macro_f1 REAL, kappa REAL, notes TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)")

def save_grades(df, db_path=DB_PATH):
    initialize_db(db_path)
    columns = ["student_id", "question_id", "answer_text", "predicted_score", "teacher_score", "confidence"]
    df = df.copy()
    for col in columns:
        if col not in df:
            df[col] = None
    with sqlite3.connect(db_path) as conn:
        df[columns].to_sql("grades", conn, if_exists="append", index=False)

def read_grades(db_path=DB_PATH):
    initialize_db(db_path)
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM grades ORDER BY id DESC", conn)

def read_experiments(db_path=DB_PATH):
    initialize_db(db_path)
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM experiments ORDER BY id DESC", conn)
