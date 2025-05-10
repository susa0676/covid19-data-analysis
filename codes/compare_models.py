from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, unix_timestamp
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor, LinearRegression, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from datetime import datetime


spark = SparkSession.builder.appName("COVID19_RegionWise_ModelComparison").getOrCreate()

file_path = "hdfs://localhost:9000/covid19_data/covid_19.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)


df = df.withColumnRenamed("Date", "date").withColumnRenamed("Confirmed", "confirmed")
df = df.withColumn("date", to_date("date", "yyyy-MM-dd"))
df = df.filter((col("date").isNotNull()) & (col("confirmed").isNotNull()) & (col("Country/Region").isNotNull()))
df = df.withColumnRenamed("Country/Region", "Country_Region")
df = df.withColumn("date", df["date"].cast("timestamp"))

min_date = df.select("date").rdd.map(lambda row: row["date"]).min()
min_date_ts = int(datetime.combine(min_date, datetime.min.time()).timestamp())
df = df.withColumn("date_num", unix_timestamp("date") - min_date_ts)


indexer = StringIndexer(inputCol="Country_Region", outputCol="region_indexed")
df = indexer.fit(df).transform(df)


assembler = VectorAssembler(inputCols=["date_num", "region_indexed"], outputCol="features")
df = assembler.transform(df).select("features", "confirmed")


train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

evaluator_rmse = RegressionEvaluator(labelCol="confirmed", metricName="rmse")
evaluator_mae = RegressionEvaluator(labelCol="confirmed", metricName="mae")
evaluator_r2 = RegressionEvaluator(labelCol="confirmed", metricName="r2")

def evaluate_model(name, predictions):
    rmse = evaluator_rmse.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)
    print(f"\n🔎 {name} Evaluation:")
    print(f"➡️  RMSE : {rmse:.2f}")
    print(f"➡️  MAE  : {mae:.2f}")
    print(f"➡️  R²   : {r2:.4f}")

dt = DecisionTreeRegressor(featuresCol="features", labelCol="confirmed", maxBins=200)
dt_model = dt.fit(train_data)
dt_predictions = dt_model.transform(test_data)
evaluate_model("Decision Tree Regressor", dt_predictions)


lr = LinearRegression(featuresCol="features", labelCol="confirmed")
lr_model = lr.fit(train_data)
lr_predictions = lr_model.transform(test_data)
evaluate_model("Linear Regression", lr_predictions)


gbt = GBTRegressor(featuresCol="features", labelCol="confirmed", maxIter=50, maxBins=200)
gbt_model = gbt.fit(train_data)
gbt_predictions = gbt_model.transform(test_data)
evaluate_model("Gradient Boosted Trees", gbt_predictions)

model_save_path = "model/gbt"
gbt_model.write().overwrite().save(model_save_path)
print("✅ GBT model saved to HDFS.")
