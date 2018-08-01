import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import os

def knn(name, neighbors, dir):
    neighbors = int(neighbors)
    name = int(name)
    file_path = os.path.join(dir, 'imageprocessing/faces.mat')

    returnvals = {}
    data = loadmat(file_path)
    array = data['faces']
    array = array.transpose(2,0,1).reshape(400,-1)

    df = pd.DataFrame(array)
    scores = np.zeros(370)

    # start indices of glasses-wearers
    # website ones are 180, 190, 260 !!
    indices = np.array([10,30,50,120,130,160,240,270,300,330])
    for i in indices:
        scores[i:i+10] = 1

    # plot a face
    # fig = plt.figure(figsize=(5,5))
    # array = np.array(df.iloc[260], dtype=np.uint8)
    # imgplot = plt.imshow(array.reshape(112, 92), cmap='gray')
    # plt.axis('off')
    # plt.show()
    # fig.savefig('C:\\Users\\Feroze\\Google Drive\\DUKE 2016-2020\\PYTHON\\django-machine-learning\\mysite\\imageprocessing\\static/images/T\'Challa.png')

    # scale the data (doesn't make too much of a difference)
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # scaler.fit(df)
    # scaled_features = scaler.transform(df)
    # df_feat = pd.DataFrame(scaled_features)

    fred = df.iloc[180]
    df.drop(df.index[[180,181,182,183,184,185,186,187,188,189,
                      190,191,192,193,194,195,196,197,198,199,
                      260,261,262,263,264,265,266,267,268,269]], inplace=True)

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(df, scores, test_size=0.3, random_state=42)

    # use KNN
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=neighbors)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)


    # if name == 1:
    #     index = 180
    # else:
    #     index = 190

    returnvals['certainty'] = knn.predict_proba(np.array(fred).reshape(1,-1))
    returnvals['certainty'] = returnvals['certainty'][0][0]*100
    # print(returnvals['certainty'])
    returnvals['prediction'] = knn.predict(np.array(fred).reshape(1,-1))

    # check results!
    from sklearn.metrics import classification_report, confusion_matrix
    returnvals['cm'] = confusion_matrix(y_test,pred)
    returnvals['cr'] = classification_report(y_test,pred)



    # checking best NN number - Will take some time!
    # error_rate = []
    # for i in range(1, 40):
    #     knn = KNeighborsClassifier(n_neighbors=i)
    #     knn.fit(X_train, y_train)
    #     pred_i = knn.predict(X_test)
    #     error_rate.append(np.mean(pred_i != y_test))
    #
    # plt.figure(figsize=(10,6))
    # plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
    #          markerfacecolor='red', markersize=10)
    # plt.title('Error Rate vs. K Value')
    # plt.xlabel('K')
    # plt.ylabel('Error Rate')
    # plt.show()

    # generate ROC curve
    from sklearn import metrics
    probs = knn.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    plt.clf()
    sns.set_style('darkgrid')
    plt.title('Receiver Operating Characteristic for Num Neighbors: %d' % neighbors)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(os.path.join(dir, 'imageprocessing/static/graphs/ROC.png'))

    return returnvals
