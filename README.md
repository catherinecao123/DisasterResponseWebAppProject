# Disaster Response Pipeline Project

### Project Overview
<p>In this course, you've learned and built on your data engineering skills to expand your opportunities and potential as a data scientist. In this project, you'll apply these skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.<br>

<p>In the Project Workspace, you'll find a data set containing real messages that were sent during disaster events. You will be creating a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.<br>

<p>Your project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. This project will show off your software skills, including your ability to create basic data pipelines and write clean, organized code!<br></p>

<p>Below are a few screenshots of the web app.</p><br>
<img src="disasterapp_example.png" alt="DisasterResponseAppExample" width="104" height="142">

### Project Components
<p>There are three components you'll need to complete for this project.</p><br>
<dl>
 <dt>1. ETL Pipeline</dt>
  <dd>In a Python script, process_data.py, write a data cleaning pipeline that:
    <ul>
Loads the messages and categories datasets
Merges the two datasets
Cleans the data
Stores it in a SQLite database
    <ul>
  </dd>
<dt>2. ML Pipeline</dt>
  <dd>In a Python script, train_classifier.py, write a machine learning pipeline that:
    <ul>
Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file
   </ul>
  </dd>
<dt>3. Flask Web App</dt>
<dd>We are providing much of the flask web app for you, but feel free to add extra features depending on your knowledge of flask, html, css and javascript. For this part, you'll need to:
  <ul>
Modify file paths for database and model as needed
Add data visualizations using Plotly in the web app.
  </ul>
</dd>
</dl>
### Instructions to run commands in Linux:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ to visulize the web app
http://view6914b2f4-3001.udacity-student-workspaces.com
