#!/bin/bash
source /spark-examples/env.sh
/usr/local/hadoop/bin/hdfs dfs -rm -r /part1/input/
/usr/local/hadoop/bin/hdfs dfs -mkdir -p /part1/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ./data/train.csv /part1/input
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ./data/test.csv /part1/input
/usr/local/spark/bin/spark-submit --master=spark://$SPARK_MASTER:7077 ./part1.py hdfs://$SPARK_MASTER:9000/part1/input/train.csv hdfs://$SPARK_MASTER:9000/part1/input/test.csv

