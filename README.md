# Implementing ID3 Algorithm From Scratch

### Files Included:
* **driver.py**:
    * Read dataset.
    * Builds decision tree by calling the DecisionTree Class.
    * Print leaf and non-leaf nodes of the Tree.
    * Implements the Reduced Error Pruning Algorithm.
* **DecisionTree.py**:
    * Implements the **ID3 Algorithm** for generating the Decision Tree.

## Requirements
* Python 3
* Command Line Interface
* Pandas * https://pandas.pydata.org/*
* Scikit Learn

## Steps to run the program
* Open the CLI and run the command, *python3 driver.py url_link headers*
* Replace the *url_link* in the above command with the url to the dataset
* Replace the *headers* in the above command with the corresponding headers of the dataset

### Example Run commands
* **Iris Dataset**: _python3 driver.py "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data" "SepalL,SepalW,PetalL,PetalW,Class"_
* **Tic-Tac-Toe Dataset**: _python3 driver.py "https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data" "top-left-square,top-middle-square,top-right-square,middle-left-square,middle-middle-square,middle-right-square,bottom-left-square,bottom-middle-square,bottom-right-square,Class"_
* **Haberman Survival Dataset**: *python3 driver.py "https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data" "Age,Patient_year_of_operation,Number_of_positive_axillary_nodes_detected,Survival_Class"*