import pyspark
from pyspark.sql import SparkSession
from faker import Faker
from pyspark.sql.functions import cast

fake = Faker()
# Initialize SparkSession
spark = SparkSession.builder \
    .appName("Fraud Detection") \
    .getOrCreate()

# Set the log level to ERROR to reduce verbosity
sc = spark.sparkContext
sc.setLogLevel("ERROR")

# Generate synthetic transaction data
num_transactions = 1000  # Number of transactions you want to generate

transactions = [
    (
        idx + 1,  # Transaction ID starts from 1
        fake.random_number(digits=4, fix_len=True) / 100,  # Random amount 
        fake.company(),  # Random merchant name
        fake.word(),  # Random category
        fake.boolean()  # Random boolean indicating whether it's fraudulent or not
    )
    for idx in range(num_transactions)
]

# Create DataFrame
data = spark.createDataFrame(transactions, ["transaction_id", "amount", "merchant", "category", "is_fraud"])
data = data.withColumn("transaction_id_str", data["transaction_id"].cast("string"))

# Show the generated data
data.show(truncate=False)

try:
    DIRECTORY = "/Fraud Detection"
    transactions_file = f"file://{DIRECTORY}/transactions.txt"

    # Use DataFrameWriter with 'mode("overwrite")'
    data.write.mode("overwrite").csv(transactions_file, sep="\t", header=True)
    print(f"Data saved successfully with {data.count()} rows!")
except Exception as e:
    print(f"Error saving data: {e}")