import os

# Assuming your SQLite database file is named users.db
db_file = 'users.db'

# Check if the file exists before attempting to delete
if os.path.exists(db_file):
    os.remove(db_file)
    print(f"Database {db_file} deleted successfully.")
else:
    print(f"Database {db_file} does not exist.")
