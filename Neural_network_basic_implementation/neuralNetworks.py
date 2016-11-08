from sklearn.datasets import load_breast_cancer

# load data
cancer = load_breast_cancer()

'''
>>> cancer.keys()
['target_names', 'data', 'target', 'DESCR', 'feature_names']
DESCR : a description of complete dataset
feature_names : an array of names of the features used in data
target_names : an array of names of targets,i.e output variabes
target : an array of target values, 0 is for malignant and 1 for benign
data : a multi-dimensional array of feature values.

>>> cancer['data'].shape
(569, 30)
It means that we have 569 data points with 30 features/
'''

X = cancer['data']
y = cancer['target']

'''
X is capital while y is small...
Because X is multi-dimensional where as y is 1-D array
'''

from sklearn.model_selection import train_test_split
'''
Splitting data and targets for testing and training data with
SciKit Learn's train_test_split function from model_selection: 
'''
X_train, X_test, y_train, y_test = train_test_split(X, y)


from sklearn.preprocessing import StandardScaler
'''
The neural network may have difficulty converging before the maximum number of iterations allowed if the data is not normalized.
Multi-layer Perceptron is sensitive to feature scaling, so it is highly recommended to scale the data
Here we use built-in StandardScaler for standardization
'''
scaler = StandardScaler()

# Fit only to the training data
scaler.fit(X_train)

# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.neural_network import MLPClassifier
'''
Importing Multi-Layer Perceptron Classifier model
'''
# creating an instance of the model
# hidden_layer_sizes : a tuple consisting of the number of neurons we want at each layer.
mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))

# fitting the training data to our model
mlp.fit(X_train,y_train)

# generating predictions
predictions = mlp.predict(X_test)



from sklearn.metrics import classification_report,confusion_matrix
'''
Now we can use SciKit-Learn's built in metrics such as a classification report and confusion matrix
to evaluate how well our model performed
'''
# generate a confusion matrix:
print("Confusion matrix:")
print(confusion_matrix(y_test,predictions))


# generate classification report:
print("\nClassification report:")
print(classification_report(y_test,predictions))


'''
TODO:
Try playing around with the number of hidden layers and neurons and see how they effect the results!

INSIGHTS:
coefs_ is a list of weight matrices, where weight matrix at index i represents the weights between layer i and layer i+1.

intercepts_ is a list of bias vectors, where the vector at index i represents the bias values added to layer i+1.

>>> mlp.coefs_[0]  # to get the weight matrix b/w layer 1 and 2
>>> mlp.intercepts_[0] # to get the bias values added to layer i+1 
'''
