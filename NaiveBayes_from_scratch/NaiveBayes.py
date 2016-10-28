'''
DTU-CS101 ML TUTORIAL
=====================
* Gaussian Naive Bayes algorithm implementation on Pima Indians Diabetes Data Set.(https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)
* Before running this script, make sure that you have downloaded and kept the "pima-indians-diabetes.data" file in same directory as this python script.
* Python 2 and 3 compatible.
'''


import csv
import random
import math


def loadCSV(filename):
    '''
    function to load CSV file
    '''
    with open(filename,"r") as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for i in range(len(dataset)):
            dataset[i] = [float(x)for x in dataset[i]]     #converting all instances in a row to float values from string
    return dataset        


def SplitDataset(dataset,split_ratio):
    '''
    function to split dataset into test data and training data according to the specified split ratio
    '''
    trainsize=int(len(dataset)*(split_ratio))
    trainset = []
    copy = list(dataset)

    while len(trainset) < trainsize:
        index= random.randrange(len(copy))
        trainset.append(copy.pop(index))
    return trainset,copy



def separateByClass(dataset):
    '''
    function to separate the passed dataset according to classvalue(0 or 1)
    A dictionary with keys 0 and 1 is created, where each key corrrespoonds to a list of rows of passed dataset.
    '''
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if vector[-1] not in separated:     # vector[-1] is the class value(0 or 1)
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated    
            

def summarize(dataset):
    '''
    function to summarize the dataset stats.
    mean and stddev of a complete column is calculated and stored as a tuple in a list.
    '''
    summaries = [(mean(attribute),stddev(attribute)) for attribute in zip(*dataset)]    #zip(*dataset) lets you access the data column-wise
    del summaries[-1]         #deleting the stats tuple for class variabe
    return summaries


def summarizeByClass(dataset):
    '''
    function to generate summaries dictonary for 0 and 1 classvalue.
    '''
    separated = separateByClass(dataset)
    summaries = {}
    for classValue,instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries



def mean(numbers):
    '''
    function to find mean of numbers present in the passed list
    '''
    return float(sum(numbers))/len(numbers)


def stddev(numbers):
    '''
    function to find 'sample standard deviation' of the numbers in the passed list
    '''
    u = mean(numbers)
    var = float(sum([(x-u)**2 for x in numbers]))/(len(numbers) - 1) 
    return math.sqrt(var)



def calculateProbability(x,mean,stdev):
    '''
    function to calculate probability using gaussian probability density function
    '''
    exponent = math.exp(-((math.pow(x-mean,2))/(2*stdev**2)))
    return (1.0/(math.sqrt(math.pi*2)*stdev)*exponent)


def classProb(summaries,input):
    '''
    function to generate class probabilities,i.e probaility with which our input set belongs to classvalue 0 or 1.
    '''
    probabilities = {}
    for classValue,classSum in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSum)):
            mean,stddev = classSum[i]
            x = input[i]
            probabilities[classValue] *= calculateProbability(x,mean,stddev)    #multiplying together the attribute probabilities.
    return probabilities        


def predict(summaries,inputVector):
    '''
    function to predict a classvalue for the passed testcase
    here we look for the largest probability and return the associated class(Label).
    '''
    prob = classProb(summaries,inputVector)
    bestLabel,bestProb = None,-1

    for classValue,probability in prob.items():
        if bestLabel is None or probability > bestProb:
            bestLabel = classValue
            bestProb = probability
    return bestLabel


def getPredictions(summaries,testset):
    '''
    function to generate predictions for our each row(instance) of our dataset
    '''
    predictions = []
    for i in range(len(testset)):
        result = predict(summaries,testset[i])
        predictions.append(result)
    return predictions    


def getAccuracy(testset,predictions):
    '''
    function to calculate accuracy by comparing actual classvalues with the predicted classvalues
    '''
    correct= 0
    for i in range(len(testset)):
        if testset[i][-1] == predictions[i]:
            correct += 1
    return (correct/float(len(predictions)))*100.0



def main():
    filename = "pima-indians-diabetes.data"   
    split_ratio = 0.67
    
    dataset = loadCSV(filename)
    trainset,testset = SplitDataset(dataset,split_ratio)
    print('Split %d rows into train = %d and test = %d rows.'%(len(dataset),len(trainset),len(testset)))
    

    summaries = summarizeByClass(trainset)
    predictions = getPredictions(summaries,testset)
    acc= getAccuracy(testset,predictions)
    print("Accuracy:%0.1f"%(acc)+"%")
    



if __name__ == "__main__":
    main()


