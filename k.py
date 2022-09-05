
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from  sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
import  numpy as np
open_=pd.read_csv("Online Retail.csv")
x=open_.head()

#process for country
country=pd.DataFrame(open_[['Country']]).to_numpy().reshape((1,len(open_)))

model=LabelEncoder()
model.fit(country[0])
country_arr=model.transform(country[0])
country_arr=np.resize(country_arr,new_shape=(len(country_arr),1))
#procss words
des=pd.DataFrame(open_[['Description']]).to_numpy()


total_des=[]
for ext in range(len(des)):

    model2=TfidfVectorizer()
    description_Arr=model2.fit_transform(des[ext]).toarray()

    total_des.append(description_Arr[0])


####
total_x_train=[]
y=pd.DataFrame(open_[['Quantity']]).to_numpy()

x=pd.DataFrame(open_[['UnitPrice']]).to_numpy()
import  numpy as np

for total_x_insert in range(len(total_des)):

    final=[]
    final.clear()
    final.append(country_arr[total_x_insert])

    final.append(x[total_x_insert])
    for x_itr1 in range(0,len(total_des[total_x_insert])):
        final.append(total_des[total_x_insert][x_itr1])
    final=np.resize(final,new_shape=(1,7))
    total_x_train.append(final)
total_x_train=np.resize(total_x_train,new_shape=(18736,7))

##########3

X_train, X_test, y_train, y_test=train_test_split(total_x_train,y,shuffle=True)



import matplotlib.pyplot as plt
from keras import layers,Sequential,losses

from sklearn.metrics import mean_squared_error
x_trans=np.resize(X_train,new_shape=(14052,3))
x_TEST=np.resize(X_test,new_shape=(14052,3))
#plt.scatter(x_trans,y_train)
#plt.show()
from keras.constraints import min_max_norm
wt=[.5,.6]
model_ml=Sequential()
model_ml.add(layers.Dense(64,activation="relu",kernel_constraint=min_max_norm(min_value=wt[0],max_value=wt[1])))
model_ml.add(layers.Dense(64,activation="relu",kernel_constraint=min_max_norm(min_value=wt[0],max_value=wt[1])))


model_ml.add(layers.Dense(1))
x_trans=np.asarray(x_trans).astype(np.int)

y_train=np.asarray(y_train).astype(np.int)
x_TEST=np.asarray(x_TEST).astype(np.int)
#RMS<Adadelta
#BinaryCrossentropy<KLDivergence<MeanAbsolutePercentageError<MeanSquaredLogarithmicError<mean_absolute_percentage_error
model_ml.compile(optimizer='adam',loss= "MSE")
model_ml.fit(x_trans,y_train,epochs=100,verbose=1)
pred = model_ml.predict(x_TEST)
first_pred=[]
for extract_ in range(4684):
    first_pred.append(pred[extract_])
score=[]
for pred_itr in range(0,30):

    print("pred {} , real : {}".format(pred[pred_itr],y_test[pred_itr]))
    score.append((pred[pred_itr]/y_test[pred_itr])*100)

print(mean_squared_error(y_true=y_test,y_pred=first_pred))