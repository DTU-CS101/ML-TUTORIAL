'''
DTU-CS101 ML TUTORIAL
=====================

* kNN model implementation on iris data set. (http://archive.ics.uci.edu/ml/datasets/Iris)
* Before running this script, make sure that you have downloaded and kept the iris.data file in same directory as this python script.
* Python 2 and 3 compatible.
'''

import csv
import random
import math
import operator



def loadDataset(filename,split_ratio,training_data,test_data):
    '''
    function to load data set.
    Make sure that the data file exists in same directory as this python script.
    '''
    with open(filename,"r") as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        
        for x in range(len(dataset)- 1):    # iterate each row of dataset
            for y in range(4):              #iterate first 4 columns of xth row
                dataset[x][y] = float(dataset[x][y])
                
            if random.random() < split_ratio:        # split data into two parts, training_data and test_data. The distribution is random and approximate ratio of split is equal to split_ratio
                training_data.append(dataset[x])
            else:
                test_data.append(dataset[x])




def euclidean_distance(ins1,ins2,features):
    '''
    function to find eucledian distance between two given instances.
    'features' is the number of features of instances to consider in finding euclidean distance.(4 in case of iris dataset)
    '''
    distance = 0
    for x in range(features):
        distance += pow(ins1[x] - ins2[x],2)

    return math.sqrt(distance)



def getNeighbours(training_data,test_ins,k):
    '''
    get k nearest neighbours of test_data instance by comparing its euclidean distance with every training_data instance.
    '''
    
    distances = []      # list to store distance of test_data instance from every training_data instance 
    features = len(test_ins) - 1   # Number of features to consider in finding euclidean distance


    for x in range(len(training_data)):                  #iterate each row of training_data 
        distance = euclidean_distance(test_ins,training_data[x], features)
        distances.append((training_data[x],distance))

    '''
    distances now contains a list of tuples where first element is the training_data row and second element is distance
    '''
    distances.sort(key = operator.itemgetter(1))  # sort distances according to second element of tuple, i.e distance.


    neighbours = []
    for x in range(k):                      #append k nearest neighbours of test_data instance in list called neighbours
        neighbours.append(distances[x][0])

    return neighbours


def getResponseVotes(neighbours):
    '''
    function to select the nearest of the selected k neighbours on the basis of their frequency
    '''
    classvotes = {}       # a dictionary to store frequency of each type of neighbour

    
    for x in range(len(neighbours)):       # iterate through each neighbour
        response = neighbours[x][-1]       # response is the last column of our dataset ('type of iris' in case of iris dataset)
        if response in classvotes:
            classvotes[response] +=1
        else:
            classvotes[response] = 1

    '''
    create a sorted list of tuples in descending order.
    In iris data set, this tuple is of form:(iris-type, frequency of iris-type)
    '''
    sortedvotes = sorted(classvotes.items(),key = operator.itemgetter(1),reverse = True)    

    return sortedvotes[0][0]     # return predicted iris-type




def getAccuracy(test_data,predictions):
    '''
    function to calculate accuracy of our kNN model.
    '''
    correct = 0     # to keep count of the iris-types which were predicted correctly by our model

    for x in range(len(test_data)):
        if test_data[x][-1] == predictions[x]:
            correct+=1
            
    return (correct/float(len(test_data)))*100.0     # return percentage accuracy   


            

def main():
    training_data = []
    test_data = []

    '''
    model specifications
    ''' 
    filename = "iris.data"       # name of data file to load
    split_ratio = 0.66           # splitting ratio of training and test data set
    k = 3                        # k is the no. of nearest neighbors we wish to take vote from


    '''
    load data set and create training and test data
    '''
    loadDataset(filename,split_ratio,training_data,test_data)

    print("Training data is:\n"  + '\n'.join(str(x) for x in training_data))
    print("\n\nTest data is:\n"  + '\n'.join(str(x) for x in test_data))


    '''
    generate predictions for each test_data instance and store them in the list, predictions.
    '''
    predictions = []    

    for x in range(len(test_data)):                       
        neighbours = getNeighbours(training_data, test_data[x],k)
        prediction = getResponseVotes(neighbours)
        predictions.append(prediction)

    '''
    determine model accuracy by comparing predictions with the actual test_data
    '''
    modelAccuracy = getAccuracy(test_data,predictions)


    print("The kNN model predicted correct iris-types with accuracy:%0.3f"%(modelAccuracy))

    return 



if __name__ == "__main__":
    main()


    
