# drop_faces_table.py
import sqlite3

def drop_faces_table():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS faces")
    conn.commit()
    conn.close()
    print("Dropped faces table.")

if __name__ == "__main__":
    drop_faces_table()
 
