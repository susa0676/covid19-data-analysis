from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, unix_timestamp
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from datetime import datetime
from pyspark.sql import Row

spark = SparkSession.builder.appName("COVID19_RegionWise_RF").getOrCreate()

file_path = "hdfs://localhost:9000/covid19_data/covid_19.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)


df = df.withColumnRenamed("Date", "date").withColumnRenamed("Confirmed", "confirmed")
df = df.withColumn("date", to_date("date", "yyyy-MM-dd"))
df = df.filter((col("date").isNotNull()) & (col("confirmed").isNotNull()) & (col("Country/Region").isNotNull()))

df = df.filter(df["date"].isNotNull())
df = df.withColumn("date", df["date"].cast("timestamp"))  
min_date = df.select("date").rdd.map(lambda row: row["date"]).min()


min_date_ts = int(datetime.combine(min_date, datetime.min.time()).timestamp())
df = df.withColumn("date_num", unix_timestamp("date") - min_date_ts)


df = df.withColumnRenamed("Country/Region", "Country_Region")

indexer = StringIndexer(inputCol="Country_Region", outputCol="region_indexed")

fitted_indexer = indexer.fit(df)  
fitted_indexer.write().overwrite().save("model/indexer")
df = fitted_indexer.transform(df)


assembler = VectorAssembler(inputCols=["date_num", "region_indexed"], outputCol="features")
df = assembler.transform(df).select("features", "confirmed")



train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)


rf = RandomForestRegressor(featuresCol="features", labelCol="confirmed",maxBins=256)


param_grid = (ParamGridBuilder()
              .addGrid(rf.numTrees, [20, 50])
              .addGrid(rf.maxDepth, [5, 10])
              .build())

evaluator = RegressionEvaluator(labelCol="confirmed", metricName="rmse")

crossval = CrossValidator(estimator=rf,
                          estimatorParamMaps=param_grid,
                          evaluator=evaluator,
                          numFolds=3)


cv_model = crossval.fit(train_data)
best_model = cv_model.bestModel


predictions = best_model.transform(test_data)
rmse = evaluator.evaluate(predictions)
mae = RegressionEvaluator(labelCol="confirmed", metricName="mae").evaluate(predictions)
r2 = RegressionEvaluator(labelCol="confirmed", metricName="r2").evaluate(predictions)

print(f" Random Forest Model Metrics:")
print(f" RMSE: {rmse}")
print(f" MAE : {mae}")
print(f" R²  : {r2}")


best_model.write().overwrite().save("model/random_forest_model")

assembler.write().overwrite().save("model/assembler")

print(" Model and transformers saved.")
