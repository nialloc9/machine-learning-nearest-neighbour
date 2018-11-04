import numpy, warnings, pandas, random
from collections import Counter


def squared(x):
    return x ** 2


def calculate_euclidean_distance(features, predict):
    return numpy.linalg.norm(numpy.array(features) - numpy.array(predict))


def k_nearest_neighbour(data, predict, k=3):
    if len(data) >= k:
        return warnings.warn("K is set to less than total voting groups")

    distances = []

    for group in data:
        for features in data[group]:
            euclidean_distance = calculate_euclidean_distance(features, predict)
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]

    vote_result = Counter(votes).most_common(1)[0][0]

    confidence = Counter(votes).most_common(1)[0][1] / k

    return vote_result, confidence


data_frame = pandas.read_csv('breast-cancer-wisconsin.data.txt')
data_frame.replace('?', -99999, inplace=True)

data_frame.drop(['id'], 1, inplace=True)

full_data = data_frame.astype(float).values.tolist()

random.shuffle(full_data)

test_size = 0.2

train_set = {2: [], 4: []}

test_set = {2: [], 4: []}

train_data = full_data[:-int(test_size * len(full_data))]

test_data = full_data[-int(test_size * len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote, confidence = k_nearest_neighbour(train_set, data, k=5)

        print("vote with confidence", vote, confidence)
        if group == vote:
            correct += 1

        total += 1

print("Accuracy: ", correct / total)

'''
    sklearn is using radius to get rid of outlyers and also using n_job of -1 so use that because it's faster.
'''