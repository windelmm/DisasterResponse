# Disaster Response Pipeline Project

## Table of Contents
1. [Description](#description)
2. [Getting Started](#getting_started)
	* [Dependencies](#dependencies)
	* [Installing](#installing)
    	* [Files](#files)
	* [Executing Program](#executing)
3. [Author](#author)
4. [Acknowledgement](#acknowledgement)


<a name="descripton"></a>
## Description

This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight.
The initial dataset contains pre-labelled tweet and messages from real-life disaster with the goal to build a Natural Language Processing tool that categorize relevant messages.

The Project is divided in three parts:

1. Data Processing, ETL Pipeline
2. Machine Learning Pipeline to classify text message in categories
3. Web App to show results. 

<a name="getting_started"></a>
## Getting Started

<a name="dependencies"></a>
### Dependencies
* Python 3.5+ (I used Python 3.7)
* Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
* Natural Language Process Libraries: NLTK
* SQLlite Database Libraqries: SQLalchemy
* Web App and Data Visualization: Flask, Plotly

<a name="installing"></a>
### Installing
Clone GIT repository:
```
git clone https://github.com/windelmm/DisasterResponse.git
```
<a name="executing"></a>

## Files
- \
	- README.md
	- ETL Pipeline Preparation.ipynb
	- ML Pipeline Preparation.ipynb
- \app
	- run.py
	- \templates
	   - go.html
	   - master.html
- \data
	- DisasterResponse.db
	- disaster_categories.csv
	- disaster_messages.csv
	- process_data.py
- \models
	- classifier.pkl
	- train_classifier.py

### Executing Program:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Author

* [Manu Windels](https://github.com/windelmm)

## Acknowledgements

* [Udacity](https://www.udacity.com/) 
* [Figure Eight](https://www.figure-eight.com/)


