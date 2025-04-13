import sqlite3
import asyncio
import logging
import os
from datetime import datetime

DATABASE_NAME = os.path.join(os.getcwd(), "users.db")
logging.info(f"Database path: {DATABASE_NAME}")

# Connect to the database and create the tables if they don't exist
async def initialize_db():
    await asyncio.sleep(0)  # Ensure it's a coroutine
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS users (
                        user_id VARCHAR(255) PRIMARY KEY
                    )""")
    cursor.execute("""CREATE TABLE IF NOT EXISTS threads (
                        thread_id VARCHAR(255) PRIMARY KEY,
                        user_id VARCHAR(255),
                        title TEXT,
                        created_at DATETIME,
                        updated_at DATETIME,
                        FOREIGN KEY (user_id) REFERENCES users (user_id)
                    )""")
    cursor.execute("""CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id VARCHAR(255),
                    role TEXT,
                    content TEXT,
                    image_id VARCHAR(255),
                    message_time DATETIME,
                    reply_to INTEGER,  -- New field to reference the message it's replying to
                    FOREIGN KEY (thread_id) REFERENCES threads (thread_id),
                    FOREIGN KEY (reply_to) REFERENCES messages (message_id)
                )""")
    logging.info("Users, Threads, and Messages tables created or already exist.")
    conn.commit()
    conn.close()

# Add a new user
def add_user(user_id):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (user_id) VALUES (?)", (user_id,))
    conn.commit()
    conn.close()

# Add a new thread
def add_thread(user_id, thread_id, title):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    current_time = datetime.utcnow()
    logging.debug(f"Adding thread with ID {thread_id} for user {user_id} with title {title}")
    cursor.execute("INSERT INTO threads (thread_id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                   (thread_id, user_id, title, current_time, current_time))
    conn.commit()
    conn.close()

# Remove a thread
def remove_thread(thread_id):
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logging.error(f"Error remove thread from database: {e}")
        return e
    

# Update the thread for a given user
def update_thread(user_id, new_thread_id):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET thread_id = ? WHERE user_id = ?", (new_thread_id, user_id))
    conn.commit()
    conn.close()

# Get user details
def get_user(user_id):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()
    return user

# Get threads for a given user
def get_threads(user_id):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    logging.debug(f"Retrieving threads for user {user_id}")
    cursor.execute("SELECT thread_id, title, created_at, updated_at FROM threads WHERE user_id = ?", (user_id,))
    threads = cursor.fetchall()
    conn.close()
    return [{"thread_id": thread[0], "title": thread[1], "created_at": thread[2], "updated_at": thread[3]} for thread in threads]

# Add a message to a thread
def add_message(thread_id, role, content, image_id=None, reply_to=None):
    """
    Adds a new message to the database.
    
    Args:
        thread_id (str): The ID of the thread.
        role (str): The role of the message (e.g., 'user', 'assistant').
        content (str): The content of the message.
        image_id (str, optional): The ID of an associated image file.
        reply_to (int, optional): The ID of the message this message is replying to.
    """
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    
    # Check if the message already exists
    cursor.execute(
        "SELECT message_id FROM messages WHERE thread_id = ? AND role = ? AND content = ? AND image_id = ? AND reply_to = ?",
        (thread_id, role, content, image_id, reply_to)
    )
    existing_message = cursor.fetchone()
    
    if existing_message:
        logging.debug(f"Message already exists with ID {existing_message[0]}")
        conn.close()
        return existing_message[0]
    
    current_time = datetime.utcnow()
    cursor.execute(
        "INSERT INTO messages (thread_id, role, content, image_id, message_time, reply_to) VALUES (?, ?, ?, ?, ?, ?)",
        (thread_id, role, content, image_id, current_time, reply_to)
    )
    cursor.execute(
        "UPDATE threads SET updated_at = ? WHERE thread_id = ?",
        (current_time, thread_id)
    )
    conn.commit()
    conn.close()
    return cursor.lastrowid

# Get messages for a thread
def get_messages(thread_id):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT message_id, role, content, image_id, reply_to FROM messages WHERE thread_id = ? ORDER BY message_time ASC", (thread_id,))
    messages = cursor.fetchall()
    conn.close()
    return messages

# Remove a user and its associated threads
def remove_user(user_id):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()


def get_latest_image_id_in_thread(thread_id):
    """
    Fetches the latest image_id associated with any message in the thread.
    
    Args:
        thread_id (str): The ID of the thread.
    
    Returns:
        str: The latest image_id in the thread, or None if no image is found.
    """
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT image_id FROM messages WHERE thread_id = ? AND image_id IS NOT NULL ORDER BY message_time DESC LIMIT 1",
        (thread_id,)
    )
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def get_latest_user_message(thread_id):
    """
    Fetches the latest user message in a thread.
    
    Args:
        thread_id (str): The ID of the thread.
    
    Returns:
        tuple: (message_id, content) of the latest user message, or None if no message is found.
    """
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT message_id, content FROM messages WHERE thread_id = ? AND role = 'user' ORDER BY message_time DESC LIMIT 1",
        (thread_id,)
    )
    latest_user_message = cursor.fetchone()
    conn.close()
    return latest_user_message


def update_message_content(message_id, new_content):
    """
    Updates the content of a message.
    
    Args:
        message_id (int): The ID of the message to update.
        new_content (str): The new content for the message.
    """
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE messages SET content = ? WHERE message_id = ?",
        (new_content, message_id)
    )
    conn.commit()
    conn.close()



    