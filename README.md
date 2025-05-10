# covid19 data analysis 
 using hadoop -bigdata analytics
# 🦠 COVID-19 Data Analysis Using Hadoop

A Big Data project that leverages the Hadoop ecosystem to analyze and predict COVID-19 trends using large-scale datasets. It demonstrates how HDFS, MapReduce, Hive, and PySpark can be integrated for scalable data storage, processing, querying, and machine learning.

---

## 📘 Abstract

The COVID-19 pandemic generated massive datasets involving daily infections, recoveries, and fatalities. Traditional systems struggle to process such large data efficiently. This project offers a scalable solution using the Hadoop ecosystem:

- **HDFS** for distributed storage  
- **MapReduce** for batch processing  
- **Hive** for structured queries  
- **PySpark (GBT)** for predictive analytics  
- **Python (Matplotlib/Seaborn)** for data visualization

---

## 👥 Team Members

- Arumai Stalin Milton (22BIT0677)  
- Sudharsanan G (22BIT0676)  
- Saravanan S (22BIT0663)  

---

## 🎯 Objectives

- Efficient COVID-19 data collection and storage using HDFS  
- Large-scale batch processing using MapReduce  
- Structured data querying using Hive  
- Insight visualization via Python libraries  
- Predictive modeling using PySpark’s Gradient Boosted Trees

---

## 📂 Dataset

- **Size**: 49,068 records  
- **Source**: Kaggle COVID-19 dataset  
- **HDFS Path**: `hdfs://localhost:9000/covid19_data/covid_19.csv`

### Key Features:
- Date  
- Country/Region  
- Confirmed Cases  
- Deaths  
- Recoveries  
- Active Cases  
- WHO Region

---

## 🛠️ Tools & Technologies

| Category            | Technology Used         |
|---------------------|--------------------------|
| Distributed Storage | Hadoop HDFS              |
| Batch Processing    | Hadoop MapReduce         |
| Structured Querying | Apache Hive              |
| ML & Prediction     | Apache Spark, PySpark GBT|
| Visualization       | Python (Matplotlib, Seaborn) |
| Programming         | Python, SQL, Java (Hadoop jobs) |

---

## 🧠 Architecture

1. **Data Collection**  
   - COVID-19 case data from open datasets (e.g., Kaggle)

2. **Storage**  
   - Raw data stored in **HDFS**  
   - Structured data accessible via **Hive tables**

3. **Processing**  
   - **MapReduce** jobs to extract daily, weekly, and region-based trends

4. **Analysis**  
   - Hive queries to analyze fatality rates, recovery trends, and peak zones

5. **Prediction**  
   - **Gradient Boosted Trees (GBT)** model using PySpark for forecasting

6. **Visualization**  
   - Python dashboards showcasing trends and predictions

---

## 📈 Machine Learning Models & Evaluation

| Model                  | RMSE        | MAE        | R² Score  |
|------------------------|-------------|------------|-----------|
| Decision Tree Regressor| 25,237.19   | 7,620.21   | 0.9536    |
| Linear Regression      | 115,867.10  | 30,149.38  | 0.0226    |
| Random Forest Regressor| 40,337.24   | 11,016.59  | 0.8815    |
| **Gradient Boosted Trees** | **15,891.17** | **3,871.61** | **0.9816** |

> GBT was chosen as the final model for its superior accuracy and predictive power.

---

## 📊 Insights & Visualizations

- **Daily/Monthly/Weekly Trends**  
- **Top Affected Countries**  
- **Case Fatality Rate Analysis**  
- **Active Cases Heatmap**  
- **Lockdown & Vaccine Effectiveness**

---

## 🧪 Sample Hive Queries

```sql
-- Total confirmed cases by country
SELECT Country, SUM(Confirmed) AS Total_Confirmed
FROM covid_data
GROUP BY Country
ORDER BY Total_Confirmed DESC;

-- Daily trend analysis
SELECT Date, SUM(Confirmed) AS Daily_Confirmed
FROM covid_data
GROUP BY Date
ORDER BY Date;
```
🔮 Future Enhancements
Real-time analytics using Apache Kafka

Time-series forecasting with LSTM models

Cloud deployment on AWS EMR or Google Cloud Dataproc

Integration with public health APIs

📄 References
Chakraborty & Ghosh (2020), Chaos, Solitons & Fractals

Rustam et al. (2020), IEEE Access

Syed & Zameer (2021), Journal of King Saud Univ.

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.