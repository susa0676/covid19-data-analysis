from pyspark.sql import SparkSession, Row
from pyspark.ml.feature import StringIndexerModel, VectorAssembler
from pyspark.ml.regression import GBTRegressionModel
from datetime import datetime


spark = SparkSession.builder.appName("COVID19_RegionWise_GBT_Predict").getOrCreate()

gbt_model = GBTRegressionModel.load("model/gbt") 
indexer = StringIndexerModel.load("model/indexer")  

min_date = datetime(2020, 1, 22)


input_date_str = "2021-08-01"
input_region = "India"


input_date = datetime.strptime(input_date_str, "%Y-%m-%d")
input_date_num = int((input_date - min_date).total_seconds())


input_df = spark.createDataFrame([Row(date_num=input_date_num, **{"Country/Region": input_region})])


input_df = input_df.withColumnRenamed("Country/Region", "Country_Region")


input_df = indexer.transform(input_df)


assembler = VectorAssembler(inputCols=["date_num", "region_indexed"], outputCol="features")
input_df = assembler.transform(input_df)


prediction = gbt_model.transform(input_df)
prediction.select("prediction").show()
