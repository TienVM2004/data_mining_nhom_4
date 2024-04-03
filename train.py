from preprocessing_data import *
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Model
from keras.layers import *
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.sequence import pad_sequences
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import optuna
from keras.preprocessing.text import Tokenizer

"""Dataset"""

class_labels = {
    0: 'business',
    1: 'health',
    2: 'politics',
    3: 'science',
    4: 'sport'
}

"""Machine Learning Model"""
#Logistic Regression Model
class LogisticRegressionModel:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.best_params = None
        self.best_model = None
        self.study = None

    def objective(self, trial):
        
        param = {
            'C': trial.suggest_float('C', 0.1, 10.0),
            'solver': trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']),
            'max_iter': trial.suggest_int('max_iter', 600, 1000)
        }

        lr_model = LogisticRegression(**param)
        # Calculate cross-validation scores
        cv_scores = cross_val_score(lr_model, self.x_train, self.y_train, cv=5)  # Change cv value as needed
        avg_cv_accuracy = cv_scores.mean()
        return avg_cv_accuracy

    def tune_model(self, n_trials=100):
        print("Tuning Logistic Regression model")
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(self.objective, n_trials=n_trials)
        self.best_params = self.study.best_params

    def train_best_model(self):
        self.best_model = LogisticRegression(**self.best_params)
        self.best_model.fit(self.x_train, self.y_train)



"""### Random Forest Model"""

class RandomForestModel:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.best_params = None
        self.best_model = None
        self.study = None

    def objective(self, trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        }

        rf_model = RandomForestClassifier(**param)
        scores = cross_val_score(rf_model, self.x_train, self.y_train, cv=5) 
        test_accuracy = scores.mean()
        return test_accuracy

    def tune_model(self, n_trials=100):
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(self.objective, n_trials=n_trials)
        self.best_params = self.study.best_params

    def train_best_model(self):
        self.best_model = RandomForestClassifier(**self.best_params)
        self.best_model.fit(self.x_train, self.y_train)

"""##### SVM"""

class SVM:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.best_params = None
        self.best_model = None
        self.study = None

    def objective(self, trial):
        param = {
            'C': trial.suggest_float('C', 0.1, 10.0),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])
        }

        svm_model = SVC(**param)
        svm_model.fit(self.x_train, self.y_train)
        y_pred = svm_model.predict(self.x_test)
        test_accuracy = accuracy_score(self.y_test, y_pred)
        return test_accuracy

    def tune_model(self, n_trials=100):
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(self.objective, n_trials=n_trials)
        self.best_params = self.study.best_params

    def train_best_model(self):
        self.best_model = SVC(**self.best_params)
        self.best_model.fit(self.x_train, self.y_train)

"""### Deep Learning

##### CNN for Text
"""


sequence_length = 500
max_features = 30000

tokenizer = Tokenizer(num_words=max_features, split=' ', oov_token='<unw>')
tokenizer.fit_on_texts(df['paragraph'].values)

embedding_dim = 300
num_filters = 100


# use a random embedding for the text
class CNN:
    def __init__(self, sequence_length, max_features, embedding_dim, num_filters):
        self.sequence_length = sequence_length
        self.max_features = max_features
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        
        self.inputs = Input(shape=(sequence_length,), dtype='int32')
        self.embedding_layer = Embedding(input_dim=max_features, output_dim=embedding_dim, input_length=sequence_length)(self.inputs)
        self.reshape = Reshape((sequence_length, embedding_dim, 1))(self.embedding_layer)
        
        self.conv_0 = Conv2D(num_filters, kernel_size=(3, embedding_dim), activation='relu')(self.reshape)
        self.conv_1 = Conv2D(num_filters, kernel_size=(4, embedding_dim), activation='relu')(self.reshape)
        self.conv_2 = Conv2D(num_filters, kernel_size=(5, embedding_dim), activation='relu')(self.reshape)
        
        self.maxpool_0 = MaxPool2D(pool_size=(sequence_length - 3 + 1, 1), strides=(1,1), padding='valid')(self.conv_0)
        self.maxpool_1 = MaxPool2D(pool_size=(sequence_length - 4 + 1, 1), strides=(1,1), padding='valid')(self.conv_1)
        self.maxpool_2 = MaxPool2D(pool_size=(sequence_length - 5 + 1, 1), strides=(1,1), padding='valid')(self.conv_2)
        
        self.concatenated_tensor = Concatenate(axis=1)([self.maxpool_0, self.maxpool_1, self.maxpool_2])
        self.flatten = Flatten()(self.concatenated_tensor)
        
        self.dropout = Dropout(0.5)(self.flatten)
        self.output = Dense(units=5, activation='softmax')(self.dropout)
        
        self.model = Model(inputs=self.inputs, outputs=self.output)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def summary(self):
        return self.model.summary()

    def fit(self, X_train, y_train, epochs, batch_size, verbose, validation_split):
        return self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose = verbose, validation_split=validation_split)

lstm_units=128
"""LSTM"""
class LSTM_Model:
    def __init__(self, sequence_length, max_features, embedding_dim):
        self.sequence_length = sequence_length
        self.max_features = max_features
        self.embedding_dim = embedding_dim
        
        # Define input layer
        self.inputs = Input(shape=(sequence_length,))
        
        # Embedding layer
        self.embedding_layer = Embedding(input_dim=max_features, output_dim=embedding_dim, input_length=sequence_length)(self.inputs)
        
        # LSTM layer
        self.lstm_layer = LSTM(units=lstm_units)(self.embedding_layer)
        
        # Dropout layer for regularization
        self.dropout = Dropout(0.5)(self.lstm_layer)
        
        # Output layer
        self.output = Dense(units=5, activation='softmax')(self.dropout)
        
        # Create model
        self.model = Model(inputs=self.inputs, outputs=self.output)
        
        # Compile model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def summary(self):
        return self.model.summary()

    def fit(self, X_train, y_train, epochs, batch_size, verbose, validation_split):
        return self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split=validation_split)


"""GRU"""
gru_units = 128
class GRUModel:
    def __init__(self, sequence_length, max_features, embedding_dim):
        self.sequence_length = sequence_length
        self.max_features = max_features
        self.embedding_dim = embedding_dim
        self.gru_units = gru_units
        
        # Define input layer
        self.inputs = Input(shape=(sequence_length,))
        
        # Embedding layer
        self.embedding_layer = Embedding(input_dim=max_features, output_dim=embedding_dim, input_length=sequence_length)(self.inputs)
        
        # GRU layer
        self.gru_layer = GRU(units=gru_units)(self.embedding_layer)
        
        # Dropout layer for regularization
        self.dropout = Dropout(0.5)(self.gru_layer)
        
        # Output layer
        self.output = Dense(units=5, activation='softmax')(self.dropout)
        
        # Create model
        self.model = Model(inputs=self.inputs, outputs=self.output)
        
        # Compile model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def summary(self):
        return self.model.summary()

    def fit(self, X_train, y_train, epochs, batch_size, verbose, validation_split):
        return self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split=validation_split)
    
"""Bi-LSTM"""
lstm_units = 128
class BidirectionalLSTM:
    def __init__(self, sequence_length, max_features, embedding_dim):
        self.sequence_length = sequence_length
        self.max_features = max_features
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        
        # Define input layer
        self.inputs = Input(shape=(sequence_length,))
        
        # Embedding layer
        self.embedding_layer = Embedding(input_dim=max_features, output_dim=embedding_dim, input_length=sequence_length)(self.inputs)
        
        # Bidirectional LSTM layer
        self.bi_lstm = Bidirectional(LSTM(units=lstm_units))(self.embedding_layer)
        
        # Output layer
        self.output = Dense(units=5, activation='softmax')(self.bi_lstm)
        
        # Create model
        self.model = Model(inputs=self.inputs, outputs=self.output)
        
        # Compile model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def summary(self):
        return self.model.summary()

    def fit(self, X_train, y_train, epochs, batch_size, verbose, validation_split):
        return self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split=validation_split)