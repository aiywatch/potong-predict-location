
# ## Import Libraries & Data
import pandas as pd
import numpy as np

import cleaning_data



def get_modellers(path):

    X, y = cleaning_data.get_X_y(path)
    
    # ## Encode data & Train/Test split
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    labelencoder = LabelEncoder()
    X[:, 0] = labelencoder.fit_transform(X[:, 0])
    
    onehotencoder = OneHotEncoder(categorical_features = [1])
    X = onehotencoder.fit_transform(X).toarray()
    X = X[:, 1:]
    
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
    
    
    # ## Machine Learning Algorithm    
    from sklearn.linear_model import Ridge
    regressor = Ridge()
    regressor.fit(X_train, y_train)
    
    y_pred = regressor.predict(X_test)
    score = regressor.score(X_test, y_test)
    return [regressor, labelencoder, onehotencoder]


def export_model(modellers, filename):
    from sklearn.externals import joblib
    joblib.dump(modellers, 'pickled-data/'+filename+'.pkl')


def save_model(filename, path):
    modellers = get_modellers(path)
    export_model(modellers, filename)


save_model('potong-1', 'data/2017-06-potong-1-new-freq.csv')
save_model('potong-2', 'data/2017-06-potong-2-new-freq.csv')
save_model('potong-2a', 'data/2017-06-potong-2a-new-freq.csv')
save_model('potong-3', 'data/2017-06-potong-3-new-freq.csv')

