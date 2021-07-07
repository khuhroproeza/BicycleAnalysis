# Bike Data Analysis

## Features

- Conversion of Json to Individual Objects for annotators
- Anaylsis of data
- Comparison between different annotators
- Understanding of data


## Tech

This Project uses the following main libraries:

- [Tensorflow] - TensorFlow is an end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools..
- [Seaborn] - Seaborn is a data visualization library built on top of matplotlib and closely integrated with pandas data structures in Python.
- [Matplotlib] -  Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.
- [Numpy] - NumPy is the fundamental package for scientific computing in Python. ... At the core of the NumPy package, is the ndarray object. 
- [Pandas] - pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language.
- [SkLearn] - Simple and efficient tools for predictive data analysis · Accessible to everybody, and reusable in various contexts · Built on NumPy, SciPy, and matplotlib



## Installation

To install all the libraries:

```sh
pip3 install -r requirements.txt
python3 bicycle.py
```
[Optional]:
To run the project individually and generate all the graphs
where reference.json and anonymized_project.json should be the pathlink for the files.
```sh
reference_import = "references.json"
project_import = "anonymized_project.json"
bike = Biycle(reference_import, project_import)
bike.process_start()
bike.generate_comparison_data()
bike.generate_average_time()
bike.generate_duration_overal()
bike.generate_adjusted_graph()
bike.generate_pie_for_answered_questions()
bike.generate_annotators_graph_count()
bike.generate_dataframe()
```
To view better Jupyter Notebook should be used which is a seperate file
The Project can also be run in Jupyter Notebook

## Analysis
-Data in this project is demonstrated in the powerpoint file which attempts to answer few of the following questions:
-How many annotators are there?
-Performance of the annotators?
-Dvisision and comparison of dataset
-Time taken by annotators


## CLassification Model
-Classification model is present in modelclassifcation Jupyter Notebook 
-Images have been pickled into array and pickle file exists as data_save.pkl









  
