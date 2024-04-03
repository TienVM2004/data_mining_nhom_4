from train import *
from preprocessing_data import *

# The number of test sample
test_size = 500
# Lấy tập kiểm tra từ 0 đến 500 đầu tiên của dữ liệu (Dữ liệu được đánh nhãn và merge vào 500 dòng đầu tiên của data)
x_test = df['word2vec'][:test_size]
y_test = df['label'][:test_size]
# Lấy tập huấn luyện từ 500 đến cuối cùng của dữ liệu (train and dev set)
x_train = df['word2vec'][test_size:]
y_train = df['label'][test_size:]
x_train = np.array([np.array(x) for x in x_train])
x_test = np.array([np.array(x) for x in x_test])

# Function used for evaluating machine learning models
def evaluate_model(best_model):
    y_pred = best_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy On Test Data:", accuracy)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xticks(ticks=np.arange(len(class_labels)) + 0.5, labels=[class_labels[i] for i in range(len(class_labels))], ha='center')
    plt.yticks(ticks=np.arange(len(class_labels)) + 0.5, labels=[class_labels[i] for i in range(len(class_labels))], va='center')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    misclassifications = cm.sum(axis=1) - np.diag(cm)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=np.arange(len(misclassifications)), y=misclassifications, hue=np.arange(len(misclassifications)), palette='Blues', legend=False)
    plt.xlabel('Class')
    plt.ylabel('Number of Misclassifications')
    plt.title('Misclassifications per Class')
    plt.xticks(ticks=np.arange(len(misclassifications)), labels=[class_labels[i] for i in range(len(misclassifications))])
    plt.show()

    max_misclass_idx = np.argmax(misclassifications)
    print("Class with the most misclassifications for Logistic Model:", max_misclass_idx)



if __name__ == "__main__":
    #Machine Learning Model
    ml_model = SVM(x_train, y_train, x_test, y_test) #Change the type of machine learning model just by replace the name of model class
    ml_model.tune_model()
    ml_model.train_best_model()
    evaluate_model(ml_model.best_model)

    # Deep Learning Model
    batch_size = 32
    X = tokenizer.texts_to_sequences(df['paragraph'].values)
    X = pad_sequences(X, sequence_length)
    y = pd.get_dummies(df['label']).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=500, shuffle=False)
    inputs = Input(shape=(sequence_length,), dtype='int32')
    dl_model = BidirectionalLSTM(sequence_length, max_features, embedding_dim) #Change the class name to switch
    dl_model.summary()
    history = dl_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, validation_split=0.2)
    predictions = dl_model.model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(np.argmax(y_test, axis=1), predicted_labels)
    print("Accuracy on Test Data:", accuracy)