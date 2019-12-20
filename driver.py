from DecisionTree import *
import pandas as pd
from sklearn import model_selection
import random
from itertools import combinations 
import sys


### for tic-tac-toe datasets use link: https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data
### for haberman survival datasets use link: https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data 
### for Iris Dataset use this link: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data

### header for IRIS dataset
'''
header = ['SepalL', 'SepalW', 'PetalL', 'PetalW', 'Class']
'''
### header for tic tac toe data    
'''
header = ['top-left-square',
'top-middle-square',
'top-right-square',
'middle-left-square',
'middle-middle-square',
'middle-right-square',
'bottom-left-square',
'bottom-middle-square',
'bottom-right-square', 
'Class']
'''
### header for haberman survival data
'''
header = ['Age',
'Patient_year_of_operation',
'Number_of_positive_axillary_nodes_detected',
'Survival_Class'
]
'''

#df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, names=header)
urlForDataset = sys.argv[1]
headerForDataset = sys.argv[2].split(',')
print(urlForDataset)
print(headerForDataset)
df = pd.read_csv(urlForDataset, header=None, names = headerForDataset)

lst = df.values.tolist()
t = build_tree(lst, header)
print("********** Initial Decision Tree ****************")
print_tree(t)
print()
print("********** Leaf nodes ****************")
leaves = getLeafNodes(t)
for leaf in leaves:
    print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))
print("Total Number of Leaf Node: ", len(leaves))
print()
print("********** Non-leaf nodes ****************")
innerNodes = getInnerNodes(t)
for inner in innerNodes:
    print("id = " + str(inner.id) + " depth =" + str(inner.depth))
print("Total number of Non-Leaf nodes: ",len(innerNodes))
trainDF, testDF = model_selection.train_test_split(df, test_size=0.2)
train = trainDF.values.tolist()
test = testDF.values.tolist()

t = build_tree(train, header)
print()
print("*************Tree before pruning*******")
print_tree(t)
accTestWithoutPruning = computeAccuracy(test, t)
updatedAccTestWithoutPruning = accTestWithoutPruning
print()
print("Accuracy on test data without pruning= " + str(accTestWithoutPruning))
acTestAfterPruning = 0
pruningNodes = [nodes.id for nodes in leaves]
setId = random.sample(pruningNodes, 1)

## TODO: You have to decide on a pruning strategy
for iterations in range(10000):
    ### randomly selecting 3 nodes from pruning  list to prune.
    setId = random.sample(pruningNodes, 1)
    t_pruned = prune_tree(t, setId)
    ("*************Tree after pruning******* ")
    acTestAfterPruning = max (acTestAfterPruning, computeAccuracy(test, t_pruned))
    if acTestAfterPruning > updatedAccTestWithoutPruning:
        print()
        ("*************Tree after pruning******* ")
        print_tree(t_pruned)
        print()
        print("node pruned: ",setId)
        print()
        print("Accuracy on test after pruning= " + str(acTestAfterPruning))
        t = t_pruned
        updatedAccTestWithoutPruning = acTestAfterPruning
        
print("Accuracy on test data without pruning= " + str(accTestWithoutPruning))      
print("Accuracy on test after pruning= " + str(acTestAfterPruning))