from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
import os
from dotenv import load_dotenv

class SnowflakeConnector:
    def __init__(self):
        load_dotenv()

        self.username = os.getenv("SNOWFLAKE_USER")
        self.password = os.getenv("SNOWFLAKE_PASSWORD")
        self.account = os.getenv("SNOWFLAKE_ACCOUNT")
        self.database = os.getenv("SNOWFLAKE_DATABASE")
        self.schema = os.getenv("SNOWFLAKE_SCHEMA")
        self.warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
        self.role = os.getenv("SNOWFLAKE_ROLE")

        self.uri = self._build_uri()
        self.engine = create_engine(self.uri)

    def _build_uri(self):
        encoded_user = quote_plus(self.username)
        encoded_pass = quote_plus(self.password)
        return (
            f"snowflake://{encoded_user}:{encoded_pass}@{self.account}/"
            f"{self.database}/{self.schema}?warehouse={self.warehouse}&role={self.role}"
        )

    def get_engine(self):
        return self.engine

    def get_uri(self):
        return self.uri

    def test_connection(self):
        with self.engine.connect() as conn:
            result = conn.execute(text("SELECT CURRENT_ACCOUNT(), CURRENT_REGION(), CURRENT_TIMESTAMP()"))
            return result.fetchall()