from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, unix_timestamp, from_unixtime
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import functions as F
from datetime import datetime
# 1. Start Spark Session
spark = SparkSession.builder.appName("COVID19_PySpark_DecisionTree").getOrCreate()

# 2. Load data from HDFS
file_path = "hdfs://localhost:9000/covid19_data/covid_19.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)

# 3. Rename and process columns
df = df.withColumnRenamed("Date", "date").withColumnRenamed("Confirmed", "confirmed")
df = df.withColumn("date", to_date("date", "yyyy-MM-dd"))
min_date_row = df.select(F.min("date")).first()
min_date = min_date_row[0]
min_date = datetime.combine(min_date, datetime.min.time())
# Add date_num column: seconds since min_date
df = df.withColumn("date_num", F.unix_timestamp("date") - F.lit(int(min_date.timestamp())))

# 4. Feature vector
assembler = VectorAssembler(inputCols=["date_num"], outputCol="features")
df = assembler.transform(df).select("features", "confirmed", "date_num")

# 5. Train-test split
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# 6. Define model
dt = DecisionTreeRegressor(featuresCol="features", labelCol="confirmed")

# 7. Hyperparameter tuning
param_grid = (ParamGridBuilder()
              .addGrid(dt.maxDepth, [5, 10])
              .addGrid(dt.minInstancesPerNode, [2, 5, 10])  # Equivalent to min_samples_split
              .addGrid(dt.minInfoGain, [0.0])               # No direct equivalent to min_samples_leaf
              .build())

crossval = CrossValidator(estimator=dt,
                          estimatorParamMaps=param_grid,
                          evaluator=RegressionEvaluator(labelCol="confirmed", metricName="rmse"),
                          numFolds=3)

# 8. Train model
cv_model = crossval.fit(train_data)
best_model = cv_model.bestModel

# 9. Evaluate
predictions = best_model.transform(test_data)
evaluator = RegressionEvaluator(labelCol="confirmed")

rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R²: {r2}")

#\