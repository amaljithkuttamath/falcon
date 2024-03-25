import pandas as pd
import sqlite3
import os
import re

def sanitize_name(name):
    # Remove any non-alphanumeric characters and replace them with underscores
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # If the name starts with a number, prepend an underscore
    if name[0].isdigit():
        name = '_' + name
    return name


def csv_to_sqlite(csv_filename, table_name):
    # Determine if the CSV file has a header
    with open(csv_filename, 'r') as file:
        first_line = file.readline().strip()
        has_header = any(c.isalpha() for c in first_line)
    # Read CSV  with or without header based on determination
    if has_header:
        df = pd.read_csv(csv_filename)
    else:
        df = pd.read_csv(csv_filename, header=None)
        df.columns = [f"col{i}" for i in range(len(df.columns))]
    
    table_name = sanitize_name(table_name)
    df.columns = [sanitize_name(col) for col in df.columns]
    # Read CSV without header and assign generic column names
    #df = pd.read_csv(csv_filename, header=None)
    #df.columns = [f"col{i}" for i in range(len(df.columns))]

    # Create SQLite database and connect to it
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()

    # Create table
    create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ("
    for col, col_type in zip(df.columns, df.dtypes):
        if 'int' in str(col_type):
            create_table_query += f"{col} INTEGER,"
        elif 'float' in str(col_type):
            create_table_query += f"{col} REAL,"
        else:
            create_table_query += f"{col} TEXT,"
    create_table_query = create_table_query[:-1] + ")"
    print()
    print(create_table_query)
    print()
    cursor.execute(create_table_query)

    # Insert data
    df.to_sql(table_name, conn, if_exists='append', index=False)

    # Commit and close connection
    conn.commit()
    conn.close()

def read_csv_without_header_and_convert_to_sqlite(csv_filename):
    table_name = os.path.splitext(os.path.basename(csv_filename))[0]
    print(table_name)
    csv_to_sqlite(csv_filename, table_name)


def convert_csv_to_sqlite(folder_path):
  csv_files = get_csv_files_in_folder(folder_path)
  for i in csv_files:
    read_csv_without_header_and_convert_to_sqlite(i)

def get_csv_files_in_folder(folder_path):
    csv_files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv') and os.path.isfile(os.path.join(folder_path, file_name)):
            csv_files.append(folder_path+file_name)
    return csv_files


def query_sqlite(sqlite_file, query):
    # Connect to SQLite database
    conn = sqlite3.connect(sqlite_file)
    cursor = conn.cursor()

    # Execute the query
    cursor.execute(query)
    rows = cursor.fetchall()

    # Print the results
    for row in rows:
        print(row)

    # Close connection
    conn.close()

sqlite_file = 'data.db'  
convert_csv_to_sqlite('falcon/data/')