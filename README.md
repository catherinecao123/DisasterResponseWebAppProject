# Disaster Response Pipeline Project

### Project Overview
<p>In this project, I've learned and built on the data engineering skills to expand my opportunities and potential as a data scientist. In this project, I'll apply these skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.<br>

<p>In the Project Workspace, there is a data set containing real messages that were sent during disaster events. I will be creating a machine learning pipeline to categorize these events so that we can send the messages to an appropriate disaster relief agency.<br>

<p>The project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. This project will show off software skills, including the ability to create basic data pipelines and write clean, organized code!<br></p>

<p>Below is a screenshot of the web app.</p><br>
<img src="disasterapp_example.png" alt="DisasterResponseAppExample">
<img src="stackbarchart.png" alt="DisasterResponseAppExample">

### Project Components
<p>There are three components that included to complete this project.</p>
<dl>
 <dt>1. ETL Pipeline</dt>
  <dd>In the data folder, a Python script, process_data.py, write a data cleaning pipeline that:
    <ul>
      <li>Loads the messages and categories datasets</li>
      <li>Merges the two datasets</li>
      <li>Cleans the data</li>
      <li>Stores it in a SQLite database</li>
    </ul>
  </dd>
<dt>2. ML Pipeline</dt>
  <dd>In the models folder, a Python script, train_classifier.py, write a machine learning pipeline that:
    <ul>
      <li>Loads data from the SQLite database</li>
      <li>Splits the dataset into training and test sets</li>
      <li>Builds a text processing and machine learning pipeline</li>
      <li>Trains and tunes a model using GridSearchCV</li>
      <li>Outputs results on the test set</li>
      <li>Exports the final model as a pickle file</li>
   </ul>
  </dd>
<dt>3. Flask Web App</dt>
<dd> Some Flask sample codes are provided from Udacity DS degree class, by add extra features depending on personal knowledge of flask, html, css and javascript. In this part, in the app folder, a python script, run.py:
  <ul>
    <li>Modify file paths for database and model as needed</li>
    <li>Add data visualizations using Plotly in the web app.</li>
  </ul>
</dd>
</dl>

### Instructions to run commands in Linux:
<dl>
  <dt>1. Run the following commands in the project's root directory to set up your database and model.</dt>
   <dd> - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`</dd>
   <dd> - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`</dd>
  <dt>2. Run the following command in the app's directory to run your web app.</dt>
    <dd>`python run.py`</dd>
  <dt>3. Go to http://0.0.0.0:3001/ to visulize the web app </dt>
    <dd>use `env | grep WORK` to check the working environment </dd>
    <dd>after running the above commands, go the site to check out the app:
        http://view6914b2f4-3001.udacity-student-workspaces.com</dd>
</dl>

