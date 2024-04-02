
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
import sys

#Build a spark context
spark = (SparkSession.builder
                .appName('Toxic Comment Classification')
                .config("spark.shuffle.service.enabled", "false")
                .config("spark.dynamicAllocation.enabled", "false")
                .getOrCreate())

# Load datasets
train_file = sys.argv[1]
test_file = sys.argv[2]

df_train = spark.read.format("csv") \
  .option("header", "true") \
  .option("delimiter", ",") \
  .option("multiline", "true") \
  .option("quote", "\"") \
  .option("escape", "\"") \
  .load(train_file)

df_test = spark.read.format("csv") \
  .option("header", "true") \
  .option("delimiter", ",") \
  .option("multiline", "true") \
  .option("quote", "\"") \
  .option("escape", "\"") \
  .load(test_file)

df_test = df_test.dropna(how='any')
df_train = df_train.dropna(how='any')

#converting to double type 
label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
for label_col in label_cols:
    df_train = df_train.withColumn(label_col, col(label_col).cast("double"))

out_cols = [i for i in df_train.columns if i not in ["id", "comment_text"]]

# Configure an ML pipeline
tokenizer = Tokenizer(inputCol="comment_text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="rawFeatures")
idf = IDF(inputCol="rawFeatures", outputCol="features")

# Build the pipeline for logistic regression and add it to the existing pipeline
pipeline = Pipeline(stages=[tokenizer, hashingTF, idf])

# Fit the pipeline to training data
print('Pipeline is being trained..')
pipeline_model = pipeline.fit(df_train)

# Cache intermediate results for better performance
df_train = df_train.cache()
df_test = df_test.cache()

# Make predictions on the test set in parallel for each class
print('Making predictions on test set')

# Initialize the DataFrame to hold the results for the test set
test_res = df_test.select('id')

for column in out_cols:
    print(column)
    # Create a new instance of LogisticRegression for each class
    lr = LogisticRegression(featuresCol="features", labelCol=column, regParam=0.1)
    
    # Fit the logistic regression model
    print("...fitting")
    pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, lr])
    lrModel = pipeline.fit(df_train)
    
    # Make predictions on the test set
    print("...predicting")
    predictions = lrModel.transform(df_test)
    
    # Extract the probability and append it to the test_res DataFrame
    print("...appending result")
    extract_prob = F.udf(lambda x: float(x[1]), T.FloatType())
    res = predictions.withColumn(column, extract_prob('probability')).select('id', column)
    
    # Join the predictions with the test_res DataFrame
    test_res = test_res.join(res, on="id")
    test_res.show(5)

spark.stop()