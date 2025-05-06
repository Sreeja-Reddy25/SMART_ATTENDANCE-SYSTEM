 # recreate_faces_table.py
import sqlite3

# Connect to the database
conn = sqlite3.connect('attendance.db')
cursor = conn.cursor()

# Drop the existing faces table
cursor.execute("DROP TABLE IF EXISTS faces")

# Create the faces table with the correct schema
cursor.execute('''
    CREATE TABLE faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        encoding BLOB,
        roll_no TEXT,
        class TEXT,
        picture BLOB
    )
''')
print("Table 'faces' created successfully.")

# Commit the changes
conn.commit()

# Close the connection
conn.close()

