# toxic-comment-classification
Project 2 - Practice developing spark programs to perform classification and prediction

Part1 - Toxic Comment Classification

This section mainly practices basic text processing using Apache Spark using the toxic comment text classification dataset. 

Data Source: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

Installation

	- HDFS 
	- Spark
	- Python
	- Python Packages
		- Pyspark
		- Pandas
  
Sometimes using "pip install" would not install Pandas into clusters and hdfs; this will resolve with "Pandas module not found." If you do encounter such a problem, try "sudo pip install pandas".

Usage:

	- Start the 3-node spark cluster based on HDFS
 
	- Unzip “Project2” and save our “Project2” folder under path “*/spark-examples/test-python/”
 
	- Navigate to our top-level “Project2” folder, it should contain one report in Word, one video file and four folders listed as “part1”, “part2”, “part3”, and “part4”.
	
 	- Navigate to the “part1” folder. Note that folder “part1” has a sub-folder named as “data”, a python file named as “part1.py”, a bash script named as “test.sh” and a readme text file. Dataset should be saved under "data".
	
 	- In folder "part1", run bash script “test.sh”
