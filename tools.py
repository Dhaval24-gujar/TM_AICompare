from pyathena import connect
from dotenv import load_dotenv
import re
from langchain.tools import tool

load_dotenv()

def validate_query_safety(query: str) -> bool:
    """Validate that the query is safe (read-only)."""
    # Remove comments and normalize whitespace
    cleaned_query = re.sub(r'--.*?\n', '', query, flags=re.MULTILINE)
    cleaned_query = re.sub(r'/\*.*?\*/', '', cleaned_query, flags=re.DOTALL)
    cleaned_query = ' '.join(cleaned_query.split()).upper()

    # Check for dangerous operations
    dangerous_keywords = [
        'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE',
        'TRUNCATE', 'REPLACE', 'MERGE', 'CALL', 'EXEC', 'EXECUTE',
        'GRANT', 'REVOKE', 'COMMIT', 'ROLLBACK', 'SAVEPOINT'
    ]

    for keyword in dangerous_keywords:
        if keyword in cleaned_query:
            return False

    # Must start with SELECT or WITH (for CTEs)
    if not (cleaned_query.strip().startswith('SELECT') or cleaned_query.strip().startswith('WITH')):
        return False

    return True

@tool
def athena_query(query: str)-> list:
    """
    Use this tool to query ASW Athena which is connected to a S3 bucket containing carbon emissions data.
        The Schema of Database is as follows:
        emissions (
        timestamp string,
        project_name string,
        run_id string,
        experiment_id string,
        duration double,
        emissions string,
        emissions_rate string,
        cpu_power double,
        gpu_power double,
        ram_power double,
        cpu_energy string,
        gpu_energy string,
        ram_energy string,
        energy_consumed string,
        water_consumed double,
        country_name string,
        country_iso_code string,
        region string,
        cloud_provider string,
        cloud_region string,
        os string,
        python_version string,
        codecarbon_version string,
        cpu_count bigint,
        cpu_model string,
        gpu_count bigint,
        gpu_model string,
        longitude double,
        latitude double,
        ram_total_size double,
        tracking_mode string,
        on_cloud string,
        pue double,
        wue double
    )
    Only SELECT statements are allowed

    Returns:
        list: List of tuples representing all remaining rows in the result set.

    Raises:
        ProgrammingError – If called before executing a query that returns results.
    """
    cursor = connect(s3_staging_dir="s3://carbon-logs-dev/athena-results/",
                     schema_name="carbon_emissions_db").cursor()
    if not validate_query_safety(query):
        raise ValueError(
            "Query not allowed. Only SELECT statements (and WITH clauses) are permitted. "
            "DROP, DELETE, UPDATE, INSERT, ALTER, CREATE, TRUNCATE, and other "
            "modification operations are not allowed."
        )

    if len(query) > 10000:
        raise ValueError("Query too long. Maximum length is 10,000 characters.")

    cursor.execute(query)
    return cursor.fetchall()

if __name__ == "__main__":
    response = athena_query("SELECT * FROM emissions LIMIT 10")
    print(type(response))
    for row in response:
        print(type(row))
        print(row)