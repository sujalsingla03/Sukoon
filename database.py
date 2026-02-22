"""
Sukoon - SQLite database module.
Handles all database operations: initialization, users, threads, and messages.
"""

import sqlite3
from datetime import datetime
from pathlib import Path

# Database file path (in project root)
DB_PATH = Path(__file__).parent / "sukoon.db"


def get_connection():
    """Open a new SQLite connection. Caller should close it or use context manager."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Return rows as dict-like objects
    return conn


def init_db():
    """
    Initialize the SQLite database and create tables if they don't exist.
    Tables: Users (supports password + Google OAuth), Threads, Messages.
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS Users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                password_hash TEXT,
                google_id TEXT UNIQUE,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS Threads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                thread_name TEXT NOT NULL DEFAULT 'New Chat',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES Users(id) ON DELETE CASCADE
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS Messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (thread_id) REFERENCES Threads(id) ON DELETE CASCADE
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_threads_user ON Threads(user_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_thread ON Messages(thread_id)")
        # Migration: add google_id, email to existing Users table if missing (run before index)
        try:
            cur.execute("ALTER TABLE Users ADD COLUMN google_id TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
        try:
            cur.execute("ALTER TABLE Users ADD COLUMN email TEXT")
        except sqlite3.OperationalError:
            pass
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_google_id ON Users(google_id) WHERE google_id IS NOT NULL")
        conn.commit()
    finally:
        conn.close()


# --- User operations ---

def create_user(username: str, password_hash: str) -> int | None:
    """Insert a new user (email/password). Returns user id or None if username already exists."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO Users (username, password_hash, google_id, email) VALUES (?, ?, NULL, NULL)",
            (username.strip(), password_hash),
        )
        conn.commit()
        return cur.lastrowid
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()


def get_user_by_username(username: str) -> dict | None:
    """Fetch user by username. Returns row dict or None."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, username, password_hash, google_id, email FROM Users WHERE username = ?",
            (username.strip(),),
        )
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_user_by_google_id(google_id: str) -> dict | None:
    """Fetch user by Google OAuth sub (google_id). Returns row dict or None."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, username, google_id, email FROM Users WHERE google_id = ?",
            (str(google_id),),
        )
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def create_or_get_google_user(google_id: str, email: str, name: str | None = None) -> int:
    """
    Find or create a user by Google ID. Returns user id.
    username = email (or name if email missing); password_hash = placeholder for OAuth-only users.
    """
    existing = get_user_by_google_id(google_id)
    if existing:
        return existing["id"]
    conn = get_connection()
    try:
        cur = conn.cursor()
        username = (email or name or f"user_{google_id[:12]}").strip()
        # Ensure unique username: if taken, append suffix
        base = username
        suffix = 0
        while True:
            try:
                cur.execute(
                    "INSERT INTO Users (username, password_hash, google_id, email) VALUES (?, ?, ?, ?)",
                    (username, "google_oauth", google_id, email or ""),
                )
                conn.commit()
                return cur.lastrowid
            except sqlite3.IntegrityError:
                conn.rollback()
                suffix += 1
                username = f"{base}_{suffix}"
    finally:
        conn.close()


# --- Thread operations ---

def create_thread(user_id: int, thread_name: str = "New Chat") -> int:
    """Create a new chat thread. Returns thread id."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO Threads (user_id, thread_name) VALUES (?, ?)",
            (user_id, thread_name[:100] if thread_name else "New Chat"),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_threads_for_user(user_id: int) -> list[dict]:
    """Fetch all threads for a user, ordered by most recent first."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, thread_name, created_at FROM Threads WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        )
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def get_thread(thread_id: int, user_id: int) -> dict | None:
    """Fetch a thread by id, ensuring it belongs to the user."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, thread_name, created_at FROM Threads WHERE id = ? AND user_id = ?",
            (thread_id, user_id),
        )
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_thread_name(thread_id: int, user_id: int, new_name: str) -> bool:
    """Rename a thread. Returns True if successful."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE Threads SET thread_name = ? WHERE id = ? AND user_id = ?",
            (new_name[:100], thread_id, user_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def delete_thread(thread_id: int, user_id: int) -> bool:
    """Delete a thread and all its messages. Returns True if deleted."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM Threads WHERE id = ? AND user_id = ?", (thread_id, user_id))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


# --- Message operations ---

def add_message(thread_id: int, role: str, content: str) -> int:
    """Insert a message into the Messages table. Returns message id."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO Messages (thread_id, role, content) VALUES (?, ?, ?)",
            (thread_id, role, content),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_messages_for_thread(thread_id: int, user_id: int) -> list[dict]:
    """
    Fetch all messages for a thread. Ensures thread belongs to user.
    Returns list of {role, content} ordered by timestamp.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT m.role, m.content FROM Messages m
            JOIN Threads t ON m.thread_id = t.id
            WHERE m.thread_id = ? AND t.user_id = ?
            ORDER BY m.timestamp ASC
            """,
            (thread_id, user_id),
        )
        return [{"role": row["role"], "content": row["content"]} for row in cur.fetchall()]
    finally:
        conn.close()
