import sqlite3

conn = sqlite3.connect("reviews.db")

conn.execute("""
CREATE TABLE IF NOT EXISTS history(
review TEXT,
prediction TEXT,
sentiment TEXT
)
""")

conn.close()

print("Database created successfully")