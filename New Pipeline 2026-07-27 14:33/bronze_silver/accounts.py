from pyspark import pipelines as dp
import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType, BooleanType, IntegerType, DateType, TimestampType, DoubleType, DecimalType

def transform_accounts_pandas(batch_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pandas transformation function for Salesforce accounts data.
    Applied to each partition in the batch.
    Accepts a pandas DataFrame and returns a transformed DataFrame.
    """
    # Convert numeric fields to appropriate types
    if 'AnnualRevenue' in batch_df.columns:
        batch_df['AnnualRevenue'] = pd.to_numeric(batch_df['AnnualRevenue'], errors='coerce')
    if 'NumberOfEmployees' in batch_df.columns:
        batch_df['NumberOfEmployees'] = pd.to_numeric(batch_df['NumberOfEmployees'], errors='coerce').astype('Int64')
    
    # Standardize string fields - trim whitespace and proper case
    string_fields = ['Id', 'Name', 'Type', 'AccountSource', 'Industry', 'Website']
    for field in string_fields:
        if field in batch_df.columns:
            batch_df[field] = batch_df[field].str.strip()
    
    # Proper case for Name field
    if 'Name' in batch_df.columns:
        batch_df['Name'] = batch_df['Name'].str.title()
    
    # Standardize address fields - trim and title case
    address_fields = [
        'BillingStreet', 'BillingCity', 'BillingState', 'BillingCountry',
        'ShippingStreet', 'ShippingCity', 'ShippingState', 'ShippingCountry'
    ]
    for field in address_fields:
        if field in batch_df.columns:
            batch_df[field] = batch_df[field].str.strip()
            if field not in ['BillingStreet', 'ShippingStreet']:  # Don't title case street addresses
                batch_df[field] = batch_df[field].str.title()
    
    # Standardize postal codes - trim and uppercase
    postal_fields = ['BillingPostalCode', 'ShippingPostalCode']
    for field in postal_fields:
        if field in batch_df.columns:
            batch_df[field] = batch_df[field].str.strip().str.upper()
    
    # Standardize phone/fax - remove whitespace
    contact_fields = ['Phone', 'Fax']
    for field in contact_fields:
        if field in batch_df.columns:
            batch_df[field] = batch_df[field].str.strip()
    
    # Add derived column: account_tier based on revenue and employees
    batch_df['account_tier'] = 'Small'
    if 'AnnualRevenue' in batch_df.columns and 'NumberOfEmployees' in batch_df.columns:
        batch_df.loc[
            (batch_df['AnnualRevenue'] >= 1000000) | (batch_df['NumberOfEmployees'] >= 50),
            'account_tier'
        ] = 'Medium'
        batch_df.loc[
            (batch_df['AnnualRevenue'] >= 10000000) | (batch_df['NumberOfEmployees'] >= 500),
            'account_tier'
        ] = 'Large'
        batch_df.loc[
            (batch_df['AnnualRevenue'] >= 100000000) | (batch_df['NumberOfEmployees'] >= 5000),
            'account_tier'
        ] = 'Enterprise'
    
    # Add derived column: has_complete_billing_address
    batch_df['has_complete_billing_address'] = False
    if all(field in batch_df.columns for field in ['BillingStreet', 'BillingCity', 'BillingState', 'BillingCountry']):
        batch_df['has_complete_billing_address'] = (
            batch_df['BillingStreet'].notna() &
            batch_df['BillingCity'].notna() &
            batch_df['BillingState'].notna() &
            batch_df['BillingCountry'].notna()
        )
    
    # Add derived column: has_complete_shipping_address
    batch_df['has_complete_shipping_address'] = False
    if all(field in batch_df.columns for field in ['ShippingStreet', 'ShippingCity', 'ShippingState', 'ShippingCountry']):
        batch_df['has_complete_shipping_address'] = (
            batch_df['ShippingStreet'].notna() &
            batch_df['ShippingCity'].notna() &
            batch_df['ShippingState'].notna() &
            batch_df['ShippingCountry'].notna()
        )
    
    # Add derived column: days_since_creation
    if 'CreatedDate' in batch_df.columns:
        batch_df['CreatedDate'] = pd.to_datetime(batch_df['CreatedDate'])
        batch_df['days_since_creation'] = (pd.Timestamp.now() - batch_df['CreatedDate']).dt.days
    
    # Add derived column: days_since_last_activity
    if 'LastActivityDate' in batch_df.columns:
        batch_df['LastActivityDate'] = pd.to_datetime(batch_df['LastActivityDate'])
        batch_df['days_since_last_activity'] = (pd.Timestamp.now() - batch_df['LastActivityDate']).dt.days
    
    # Add processing timestamp
    batch_df['silver_processed_at'] = pd.Timestamp.now()
    
    return batch_df

@dp.table(
    name="retail_q.retail_silver.accounts",
    comment="Silver layer Salesforce accounts with standardization and derived business metrics",
    cluster_by=["Industry", "Type"]
)
@dp.expect_or_fail("valid_account_id", "Id IS NOT NULL AND Id != ''")
@dp.expect_or_drop("valid_account_name", "Name IS NOT NULL AND Name != ''")
@dp.expect_or_drop("not_deleted", "IsDeleted = false")
@dp.expect("has_owner", "OwnerId IS NOT NULL")
@dp.expect("recent_creation", "CreatedDate >= '2020-01-01'")
def accounts():
    """
    Silver layer transformation for Salesforce accounts using pandas.
    Uses mapInPandas to process each partition with pandas operations.
    
    Transformations:
    - Filters out deleted accounts and gets current records (SCD Type 2)
    - Data quality checks on key fields (Id, Name, IsDeleted)
    - Standardization (trim strings, proper casing, address formatting)
    - Derived columns: account_tier, address completeness, activity metrics
    - Reads current snapshot from bronze SCD Type 2 table
    """
    # Define output schema matching all input fields plus derived fields
    output_schema = StructType([
        StructField("Id", StringType(), True),
        StructField("IsDeleted", BooleanType(), True),
        StructField("MasterRecordId", StringType(), True),
        StructField("Name", StringType(), True),
        StructField("Type", StringType(), True),
        StructField("ParentId", StringType(), True),
        StructField("BillingStreet", StringType(), True),
        StructField("BillingCity", StringType(), True),
        StructField("BillingState", StringType(), True),
        StructField("BillingPostalCode", StringType(), True),
        StructField("BillingCountry", StringType(), True),
        StructField("BillingStateCode", StringType(), True),
        StructField("BillingCountryCode", StringType(), True),
        StructField("BillingLatitude", DoubleType(), True),
        StructField("BillingLongitude", DoubleType(), True),
        StructField("BillingGeocodeAccuracy", StringType(), True),
        StructField("ShippingStreet", StringType(), True),
        StructField("ShippingCity", StringType(), True),
        StructField("ShippingState", StringType(), True),
        StructField("ShippingPostalCode", StringType(), True),
        StructField("ShippingCountry", StringType(), True),
        StructField("ShippingStateCode", StringType(), True),
        StructField("ShippingCountryCode", StringType(), True),
        StructField("ShippingLatitude", DoubleType(), True),
        StructField("ShippingLongitude", DoubleType(), True),
        StructField("ShippingGeocodeAccuracy", StringType(), True),
        StructField("Phone", StringType(), True),
        StructField("Fax", StringType(), True),
        StructField("Website", StringType(), True),
        StructField("PhotoUrl", StringType(), True),
        StructField("Industry", StringType(), True),
        StructField("AnnualRevenue", DoubleType(), True),
        StructField("NumberOfEmployees", IntegerType(), True),
        StructField("Description", StringType(), True),
        StructField("OwnerId", StringType(), True),
        StructField("CreatedDate", TimestampType(), True),
        StructField("CreatedById", StringType(), True),
        StructField("LastModifiedDate", TimestampType(), True),
        StructField("LastModifiedById", StringType(), True),
        StructField("SystemModstamp", TimestampType(), True),
        StructField("LastActivityDate", DateType(), True),
        StructField("LastViewedDate", TimestampType(), True),
        StructField("LastReferencedDate", TimestampType(), True),
        StructField("IsCustomerPortal", BooleanType(), True),
        StructField("Jigsaw", StringType(), True),
        StructField("JigsawCompanyId", StringType(), True),
        StructField("AccountSource", StringType(), True),
        StructField("SicDesc", StringType(), True),
        StructField("IsBuyer", BooleanType(), True),
        # Derived columns
        StructField("account_tier", StringType(), True),
        StructField("has_complete_billing_address", BooleanType(), True),
        StructField("has_complete_shipping_address", BooleanType(), True),
        StructField("days_since_creation", IntegerType(), True),
        StructField("days_since_last_activity", IntegerType(), True),
        StructField("silver_processed_at", TimestampType(), True)
    ])
    
    # Wrapper to convert simple function to iterator pattern for mapInPandas
    def transform_iterator(batch_iterator):
        for batch_df in batch_iterator:
            yield transform_accounts_pandas(batch_df)
    
    return (
        spark.readStream.table("retail_q.salesforce_bronze.account")
        .filter("__END_AT IS NULL")  # Get only current records from SCD Type 2 table
        .drop("__START_AT", "__END_AT")  # Drop SCD tracking columns
        .mapInPandas(transform_iterator, output_schema)
    )