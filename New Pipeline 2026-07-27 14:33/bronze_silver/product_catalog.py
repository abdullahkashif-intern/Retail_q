from pyspark import pipelines as dp
import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType, FloatType, DateType, IntegerType, BooleanType, TimestampType

def transform_product_catalog_pandas(batch_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pandas transformation function for product catalog data.
    Applied to each partition in the batch.
    """
    # Convert decimal to float for numeric operations
    
    batch_df['unit_price'] = batch_df['unit_price'].astype(float)
    # Standardize text fields - trim whitespace and proper case
    batch_df['product_id'] = batch_df['product_id'].str.strip()
    batch_df['product_name'] = batch_df['product_name'].str.strip().str.title()
    batch_df['category'] = batch_df['category'].str.strip().str.upper()
    batch_df['subcategory'] = batch_df['subcategory'].str.strip().str.title()
    batch_df['brand'] = batch_df['brand'].str.strip().str.title()
    batch_df['supplier_name'] = batch_df['supplier_name'].str.strip().str.title()
    
    # Add derived column: price_tier
    batch_df['price_tier'] = 'Standard'
    batch_df.loc[batch_df['unit_price'] < 10, 'price_tier'] = 'Budget'
    batch_df.loc[(batch_df['unit_price'] >= 10) & (batch_df['unit_price'] < 50), 'price_tier'] = 'Standard'
    batch_df.loc[(batch_df['unit_price'] >= 50) & (batch_df['unit_price'] < 200), 'price_tier'] = 'Premium'
    batch_df.loc[batch_df['unit_price'] >= 200, 'price_tier'] = 'Luxury'
    
    # Add derived column: days_since_launch
    batch_df['launch_date'] = pd.to_datetime(batch_df['launch_date'])
    batch_df['days_since_launch'] = (pd.Timestamp.now() - batch_df['launch_date']).dt.days
    
    # Add processing timestamp
    batch_df['silver_processed_at'] = pd.Timestamp.now()
    
    return batch_df

@dp.materialized_view(
    name="retail_q.retail_silver.product_catalog_v2",
    comment="Silver layer product catalog with data quality checks and standardization",
    cluster_by=["category", "subcategory"]
)
@dp.expect_or_fail("valid_product_id", "product_id IS NOT NULL")
@dp.expect_or_drop("valid_product_name", "product_name IS NOT NULL")
@dp.expect_or_drop("valid_unit_price", "unit_price > 0")
@dp.expect("active_products", "is_active = true")
@dp.expect("recent_launch", "launch_date >= '2020-01-01'")
def silver_product_catalog():
    """
    Silver layer transformation for product catalog using pandas.
    Uses mapInPandas to process each partition with pandas operations.
    
    Transformations:
    - Data quality checks on key fields
    - Standardization (trim strings, proper casing)
    - Derived columns: price_tier, days_since_launch
    - Reads current snapshot from bronze SCD Type 2 table
    """
    # Define output schema
    output_schema = StructType([
        StructField("product_id", StringType(), True),
        StructField("product_name", StringType(), True),
        StructField("category", StringType(), True),
        StructField("subcategory", StringType(), True),
        StructField("brand", StringType(), True),
        StructField("unit_price", FloatType(), True),
        StructField("price_tier", StringType(), True),
        StructField("supplier_name", StringType(), True),
        StructField("launch_date", DateType(), True),
        StructField("days_since_launch", IntegerType(), True),
        StructField("is_active", BooleanType(), True),
        StructField("updated_at", TimestampType(), True),
        StructField("silver_processed_at", TimestampType(), True)
    ])
    
    # Wrapper to convert simple function to iterator pattern for mapInPandas
    def transform_iterator(batch_iterator):
        for batch_df in batch_iterator:
            yield transform_product_catalog_pandas(batch_df)
    
    return (
        spark.read.table("retail_q.postgres_bronze.product_catalog")
        .filter("__END_AT IS NULL")  # Get only current records from SCD Type 2 table
        .drop("__START_AT", "__END_AT")  # Drop SCD tracking columns
        .mapInPandas(transform_iterator, output_schema)
    )