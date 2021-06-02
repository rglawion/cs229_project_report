# CS229 Project Report

This repository provides the code to replicate the cs229 report "Quantitative Trading with Machine Learning" by René M. Glawion (rglawion@stanford.edu).


## Usage of Python files

1. getDataFrankfurt.py downloads the data from yahoo finance, where we use the information provided by the Frankfurt Stock exchange.
2. createDatasetFrankfurt.py creates the dataset and features.
3. mergeDataFankfurt.py merges the files with the macroeconomic data.
4. produceMainResult.py produces the main results and saves them to parquet databases.
5. analyzeResults.py reads in the results and creates the tables we present in the paper.
6. tradingStrategy.py performs the trading strategy described in the paper.

The code for the main result runs about ~36 hours on a Ryzen Threadripper 3990X. Hence, for convenience, we also provide the output files.

Further, we provide the files analyseDataFrankfurt_ToReplicateMilestoneReport.py and analyseDataFrankfurtOverTime_ToReplicateMilestoneReport.py to replicate the results presented in the Milestone report.

Note that you have to change all paths at the beginning of each file to run the code.


## Data

We already provide all data as excel files where applicable, and for larger files, we saved everything as .parquet database files.

## Results

We also provide all results used in the analysis in the results folder.
