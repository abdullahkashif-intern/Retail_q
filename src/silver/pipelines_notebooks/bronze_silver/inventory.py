from pyspark import pipelines as dp
import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType, BooleanType, DoubleType

def transform_inventory_pandas(batch_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pandas transformation function for inventory data.
    Applied to each micro-batch in the stream.
    """
    # Standardize string fields - trim whitespace
    batch_df['inventory_id'] = batch_df['inventory_id'].str.strip()
    batch_df['product_id'] = batch_df['product_id'].str.strip()
    batch_df['store_id'] = batch_df['store_id'].str.strip()
    batch_df['warehouse_location'] = batch_df['warehouse_location'].str.strip()
    
    # Add derived column: needs_reorder
    batch_df['needs_reorder'] = batch_df['stock_quantity'] <= batch_df['reorder_level']
    
    # Add derived column: stock_status
    batch_df['stock_status'] = 'NORMAL'
    batch_df.loc[batch_df['stock_quantity'] == 0, 'stock_status'] = 'OUT_OF_STOCK'
    batch_df.loc[(batch_df['stock_quantity'] > 0) & (batch_df['stock_quantity'] <= batch_df['reorder_level']), 'stock_status'] = 'LOW_STOCK'
    batch_df.loc[batch_df['stock_quantity'] > batch_df['reorder_level'] * 3, 'stock_status'] = 'OVERSTOCKED'
    
    # Add derived column: stock_health_pct
    batch_df['stock_health_pct'] = ((batch_df['stock_quantity'] / batch_df['reorder_level']) * 100).round(2)
    
    # Add processing timestamp
    batch_df['silver_processed_at'] = pd.Timestamp.now()
    
    return batch_df

@dp.table(
    name="retail_q.retail_silver.inventory",
    comment="Silver layer inventory data with data quality checks and derived columns"
)
@dp.expect_or_drop("valid_inventory_id", "inventory_id IS NOT NULL AND inventory_id != ''")
@dp.expect_or_drop("valid_product_id", "product_id IS NOT NULL AND product_id != ''")
@dp.expect_or_drop("valid_store_id", "store_id IS NOT NULL AND store_id != ''")
@dp.expect_or_drop("valid_stock_quantity", "stock_quantity >= 0")
@dp.expect_or_drop("valid_reorder_level", "reorder_level >= 0")
@dp.expect("recent_update", "last_stock_update >= date_sub(current_timestamp(), 365)")
def inventory():
    """
    Silver layer transformation for inventory data using pandas.
    Uses mapInPandas to process each micro-batch with pandas operations.
    
    Transformations:
    - Data quality checks on key fields
    - Standardization (trim strings, handle nulls)
    - Derived columns: stock_status, needs_reorder, stock_health
    """
    # Define output schema
    output_schema = StructType([
        StructField("inventory_id", StringType(), True),
        StructField("product_id", StringType(), True),
        StructField("store_id", StringType(), True),
        StructField("stock_quantity", IntegerType(), True),
        StructField("reorder_level", IntegerType(), True),
        StructField("warehouse_location", StringType(), True),
        StructField("last_stock_update", TimestampType(), True),
        StructField("needs_reorder", BooleanType(), True),
        StructField("stock_status", StringType(), True),
        StructField("stock_health_pct", DoubleType(), True),
        StructField("silver_processed_at", TimestampType(), True)
    ])
    
    # Wrapper to convert simple function to iterator pattern for mapInPandas
    def transform_iterator(batch_iterator):
        for batch_df in batch_iterator:
            yield transform_inventory_pandas(batch_df)
    
    return (
        spark.readStream.table("retail_q.postgres_bronze.inventory")
        .mapInPandas(transform_iterator, output_schema)
    )