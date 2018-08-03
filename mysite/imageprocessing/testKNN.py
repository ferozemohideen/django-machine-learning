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
    scores = np.zeros(400)

    # start indices of glasses-wearers
    # website ones are 180, 190, 260 !!
    indices = np.array([10,30,50,120,130,160,180, 190, 260, 270, 280,300,330, 360])
    for i in indices:
        scores[i:i+10] = 1

    # plot a face
    # fig = plt.figure(figsize=(5,5))
    # array = np.array(df.iloc[260], dtype=np.uint8)
    # imgplot = plt.imshow(array.reshape(112, 92), cmap='gray')
    # plt.axis('off')
    # plt.show()

    # scale the data (doesn't make too much of a difference)
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # scaler.fit(df)
    # scaled_features = scaler.transform(df)
    # df_feat = pd.DataFrame(scaled_features)

    fred = df.iloc[180]
    johannes = df.iloc[190]
    jimothy = df.iloc[0]
    lana = df.iloc[349]

    df.drop(df.index[[180,190, 0, 349]], inplace=True)
    scores = np.delete(scores, [180, 190, 0, 349])

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(df, scores, test_size=0.3, random_state=42)

    # use KNN
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=neighbors)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)


    if name == 1:
        person = fred
    elif name == 2:
        person = johannes
    elif name == 3:
        person = jimothy
    elif name == 4:
        person = lana

    returnvals['certainty'] = knn.predict_proba(np.array(person).reshape(1,-1))

    if name == 1 or name == 2:
        returnvals['certainty'] = returnvals['certainty'][0][1]*100
    else:
        returnvals['certainty'] = returnvals['certainty'][0][0] * 100
    # print(returnvals['certainty'])
    returnvals['prediction'] = knn.predict(np.array(person).reshape(1,-1))

    # check results!
    from sklearn.metrics import classification_report, confusion_matrix
    returnvals['cm'] = confusion_matrix(y_test,pred)
    returnvals['cm1'] = returnvals['cm'][0][0]
    returnvals['cm2'] = returnvals['cm'][0][1]
    returnvals['cm3'] = returnvals['cm'][1][0]
    returnvals['cm4'] = returnvals['cm'][1][1]

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
