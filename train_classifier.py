import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = data_dict['data']
labels = data_dict['labels']

max_length = max(len(sample) for sample in data)

padded_data = np.array([sample + [0] * (max_length - len(sample)) for sample in data])

labels = np.array(labels)

x_train, x_test, y_train, y_test = train_test_split(padded_data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_test, y_predict)

# print('{}% of samples were classified correctly'.format(score * 100))

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)