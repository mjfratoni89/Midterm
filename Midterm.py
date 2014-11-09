# Michael Fratoni   Midterm
import numpy as np
import os
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

trn = os.path.expanduser('~Thunder-Dumpling/Berkeley/2014 Fall/263N/Midterm/midterm_train.csv')
test = os.path.expanduser('~Thunder-Dumpling/Berkeley/2014 Fall/263N/Midterm/midterm_test.csv')

train_data = np.genfromtxt(trn, delimiter=',', skip_header=1, usecols=(1,2,3,5,6,7,9,12))
np.random.shuffle(train_data)
#use 1,2,3,5,6,7,8,11 for best DT, only 1,2,3,5,6,7 for best kNN
col=7
Xtrain1 = train_data[:10000,:col] #train data is a randomly chosen 10000
count_train1 = train_data[:10000,-1:]
Xtrain2 = train_data[10000:,:col] #test data is everything after the random 10000
count_train_actual = train_data[10000:,-1:]

d = 10 #guess and test optimal depth for Decision Tree
DT = DecisionTreeRegressor(max_depth=d)
DT.fit(Xtrain1,count_train1)
DT_counts = DT.predict(Xtrain2)

k = 10 #guess and test optimal k for kNN
kNN = KNeighborsRegressor(n_neighbors=k)
kNN.fit(Xtrain1,count_train1)
kNN_counts = kNN.predict(Xtrain2)

#calculating RMSLE
jawn = 0
stuff = 0
n = 0
for i in range(len(count_train_actual)):
    jawn = jawn+((np.log(kNN_counts[i]+1))-(np.log(count_train_actual[i]+1)))**2
    stuff = stuff+((np.log(DT_counts[i]+1))-(np.log(count_train_actual[i]+1)))**2
    n = n+1

RMSLE_DT = np.sqrt((1/n)*stuff)
RMSLE_kNN = np.sqrt((1/n)*jawn)

print(RMSLE_DT, 'for Decision Tree')    
print(RMSLE_kNN, 'for k-Nearest Neighbor')    

test_data = np.genfromtxt(test, delimiter=',', skiprows=1, usecols=(0,1,2,3,5,6,7,9))
Xtest = test_data[:,1:col+1]

if RMSLE_DT<RMSLE_kNN:
    test_counts = DT.predict(Xtest)
    RMSLE = RMSLE_DT
else:
    test_counts = kNN.predict(Xtest)
    RMSLE = RMSLE_kNN
    
print(RMSLE)
#write values in csv file
day=0
month=8
predicted_values=os.path.expanduser('~Thunder-Dumpling/Berkeley/2014 Fall/263N/Midterm/midterm_predictions.csv')
with open(predicted_values,'w') as f:
    f.write("date,hour,count\n")
    for i in range(len(test_counts)):
        if test_data[i][3]==0:
            day=int(day+1)
            if day==32:
                day=1
                month=int(month+4)
        f.write("%s/%s/12,%f,%f\n" % (str(month),str(day),test_data[i][3],test_counts[i]))
f.close()
print(day)
