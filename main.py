import numpy, pandas, pickle
from os import path
from sklearn import neighbors, preprocessing, model_selection

'''
    Description:
        Serializes a an object into a pickle file
'''


def serialize(uri, obj):
    with open(uri, "wb") as file:
        pickle.dump(obj, file)


'''
    Description: 
        reads a pickle file
'''


def read_from_pickle(uri):
    pickle_in = open(uri, "rb")
    return pickle.load(pickle_in)


'''
    Description:
        gets a trained classifier from file or creates a new one    
'''


def get_trained_classifier(uri, features_to_train, labels_to_train):
    if path.isfile(uri):
        return read_from_pickle(uri)
    else:
        clf = neighbors.KNeighborsClassifier()
        clf.fit(features_to_train, labels_to_train)
        serialize(uri, clf)
        return clf


'''
    Description:
        normalizes data so that it is not influenced by its index        
'''


def normalize(features_to_process):
    return preprocessing.scale(features_to_process)


# fetch
data_frame = pandas.read_csv('breast-cancer-wisconsin.data.txt')


# organize
data_frame.replace("?", -99999, inplace=True)

data_frame.drop(['id'], 1, inplace=True)

features = numpy.array(data_frame.drop(['class'], 1))

labels = numpy.array(data_frame['class'])

# train
features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels, test_size=0.2)

classifier = get_trained_classifier('breast_cancer_pickle.pickle', features_train, labels_train)

# test
accuracy = classifier.score(features_test, labels_test)

print("Accuracy: ", accuracy)

# predict

test_data = numpy.array([[4, 2, 1, 1, 1, 2, 3, 2,  1], [4, 2, 1, 2, 2, 2, 3, 2,  1]])

test_data = test_data.reshape(len(test_data), -1)

predict = classifier.predict(test_data)

print("Prediction: ", predict)