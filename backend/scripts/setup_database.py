import pandas as pd
import os
from dotenv import load_dotenv
import snowflake.connector

cleaned = pd.read_csv('./Data/cleaned_data.csv')
load_dotenv()

user = os.getenv("SNOWFLAKE_USER")
password = os.getenv("SNOWFLAKE_PASSWORD")
account = os.getenv("SNOWFLAKE_ACCOUNT")
warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
role = os.getenv("SNOWFLAKE_ROLE")
database = os.getenv("SNOWFLAKE_DATABASE")
schema = os.getenv("SNOWFLAKE_SCHEMA")
table = os.getenv("SNOWFLAKE_TABLE")

conn = snowflake.connector.connect(
    user=user,
    password=password,
    account=account,
    role=role
)
cursor = conn.cursor()

# Create Warehouse
cursor.execute(f"""
    CREATE OR REPLACE WAREHOUSE {warehouse}
    WITH WAREHOUSE_SIZE = 'XSMALL'
    AUTO_SUSPEND = 300
    AUTO_RESUME = TRUE
""")
print(f"Warehouse '{warehouse}' created.")

# Create Database
cursor.execute(f"CREATE OR REPLACE DATABASE {database}")
print(f"Database '{database}' created.")

# Create Schema
cursor.execute(f"CREATE OR REPLACE SCHEMA {database}.{schema}")
print(f"Schema '{schema}' created in database '{database}'.")

# Map pandas dtypes to SQL types
def map_dtype_to_sql(dtype):
    if pd.api.types.is_integer_dtype(dtype):
        return 'INT'
    elif pd.api.types.is_float_dtype(dtype):
        return 'FLOAT'
    elif pd.api.types.is_bool_dtype(dtype):
        return 'BOOLEAN'
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return 'DATETIME'
    else:
        return 'VARCHAR(255)'

# Generate CREATE TABLE SQL query without backticks
def generate_create_table_sql(df, table_name):
    sql_lines = []
    for col in df.columns:
        col_clean = col.strip().replace(" ", "_").replace("-", "_").replace("__", "_").lower()
        sql_type = map_dtype_to_sql(df[col].dtype)
        sql_lines.append(f"  {col_clean} {sql_type}")
    sql_body = ",\n".join(sql_lines)
    return f"CREATE TABLE {table_name} (\n{sql_body}\n);"


create_table_query = generate_create_table_sql(cleaned, table)
# print(create_table_query)

cursor.execute(create_table_query)
print(f"Table {table} created or replaced.")

csv_file_path = "./Data/cleaned_data.csv"
# Upload file to the table stage
cursor.execute(f"PUT file://{csv_file_path} @%{table} AUTO_COMPRESS=TRUE")
print(f"File uploaded to table stage @{table}")

table = os.getenv("SNOWFLAKE_TABLE")
file_name = os.path.basename(csv_file_path)
cursor.execute(f"""
COPY INTO {table}
FROM @%{table}/{file_name}.gz
FILE_FORMAT = (
  TYPE = 'CSV'
  SKIP_HEADER = 1
  FIELD_OPTIONALLY_ENCLOSED_BY = '"'
)
ON_ERROR = 'ABORT_STATEMENT';
""")

print(f"Data loaded into '{table}' table.")
cursor.close()
conn.close()