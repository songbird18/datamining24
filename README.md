# Data Mining 2024 Group Project

This program was written for Data Mining Fall 2024.
It is our semester project.
The application is designed to use two different kinds
of regression to predict exam performance based on
student life factors which bear a high weight on the
student's eventual performance.
Lasso, ridge, and polynomial regression are all
implemented as part of this project. See our full 
report for more details.

## AUTHORS
Haoyang Cui - haoyang@ou.edu
Oluwademilade Jooda - oluwademilade.o.jooda-1@ou.edu
Chelsea Murray – chelsea.g.murray-1@ou.edu
Sam Bird - sam.bird@ou.edu

# UI Interactions

## HOME/GETTING STARTED

All of the program functions are listed in tabs along the
top of the window. You can navigate to a different menu by
clicking the corresponding tab along the top.

When you launch the program, the models are not yet trained.
It is HIGHLY recommended to import a dataset of at least
several hundred rows to ensure the models make accurate
predictions. However, it is also possible to manually enter
individual students for the initial model training.

Until the models have been trained, you will not be able to
make any predictions (the program will not allow it).

## UPLOAD NEW DATA

This function will REPLACE your current data with a new set
and retrain all models using the new data.

If you are not seeing your dataset in the file dialog, make
sure it is in .csv format. No other filetypes are compatible
at this time.

It is HIGHLY recommended to import a dataset of at least
several hundred rows to ensure the models make accurate
predictions.

If the popup window is asking you to wait while the models
are trained, please be patient. Depending on the dataset
size, there may be an extensive wait time to process data
and train the models.

## ADD A STUDENT

This function will allow you to input a single student 
record to add it to the existing dataset. Then, the models 
are retrained on the new set.

Please ensure you do fill out all fields. The program 
currently does not compensate for missing values.

If the popup window is asking you to wait while the models
are trained, please be patient. Depending on the dataset
size, there may be an extensive wait time to process data
and train the models.

## ADD MANY STUDENTS

This function will ADD a new set of data to your current
set and retrain all models using the combined data.

If you are not seeing your dataset in the file dialog, make
sure it is in .csv format. No other filetypes are compatible
at this time.

If the popup window is asking you to wait while the models
are trained, please be patient. Depending on the dataset
size, there may be an extensive wait time to process data
and train the models.

## PREDICT A SCORE

This function will allow you to input a single student 
record and predict their exam scores with each of the models.
(This is ideal for determining the trajectory of a single
student's academic performance.)

Please ensure you do fill out all fields. The program 
currently does not compensate for missing values.

If you receive an error message about untrained models,
please import a dataset or a series of students for training
and try again.

## PREDICT MANY SCORES

This function will import a dataset, predict scores based on
the whole set, and then present the exam score distributions
from each model's prediction results in a popup window.
(This is ideal for determining the expected performance of a
body of students collectively.)

If you are not seeing your dataset in the file dialog, make
sure it is in .csv format. No other filetypes are compatible
at this time.

If you receive an error message about untrained models,
please import a dataset or a series of students for training
and try again.
