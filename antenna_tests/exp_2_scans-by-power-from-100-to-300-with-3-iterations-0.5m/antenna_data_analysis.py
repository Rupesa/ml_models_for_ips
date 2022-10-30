from dis import dis
from operator import indexOf
import string
import sys
import glob
from tokenize import String
from collections import defaultdict
from turtle import shape
import matplotlib.pyplot as plt
import csv
import pandas as pd
import pickle
from sklearn import neighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error 
from math import sqrt
from sklearn.feature_selection import SelectKBest, f_regression, r_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 1648734594.331922, sdrt-health/mac:02010b41cdec/antenna/tag/e28068940000400386621d4f, 82

def nested_dict(n, type):
    if n == 1:
        return defaultdict(type)
    else:
        return defaultdict(lambda: nested_dict(n-1, type))

data_dir_name = ""
# data_dir_name = "scans-by-power-from-100-to-300-with-10-iterations-0.25m"
# data_dir_name = "scans-by-power-with-10-iterations-2-tags-0.50m"
return_dict = {}
return_dict = nested_dict(5, float)
data_files = list(i.split("/")[-1] for i in glob.glob(data_dir_name+"*/*"))
# print(len(data_files))
# print("")
for data_file in data_files:
    file_lst = data_file.replace(".csv", "").split("_")
    dist = file_lst[1]
    pwr = file_lst[3]
    iter = file_lst[-1]
    # print(dist, pwr, iter)

    f = open(f'{data_dir_name}{dist}_meters/{data_file}', 'r')

    tags = dict()
    ver = False

    cont = 0
    prev_timestamp = -1
    dif_sum = 0
    tag = "e28068940000400386621d4f"

    for line in f:
        if 'antenna/tag/e2' in line:
            
            vl = line.split(", ")
            
            if(len(vl) == 3):

                ver = True
                
                tag = vl[1].split('/')[-1]
                rssi = int(vl[2])

                timestamp = float(vl[0])
                if prev_timestamp == -1:
                    prev_timestamp = float(vl[0])
                dif_sum += timestamp - prev_timestamp
                prev_timestamp = timestamp
                cont+=1

                if tag not in tags:
                    tags[tag] = []

                tags[tag].append(rssi)

    if not ver:
        return_dict[tag][dist][pwr][iter]["avg_time_dif"] = 0
        return_dict[tag][dist][pwr][iter]["activations"] = 0
        return_dict[tag][dist][pwr][iter]["rssi_min"] = 170
        return_dict[tag][dist][pwr][iter]["rssi_max"] = 170
        return_dict[tag][dist][pwr][iter]["rssi_avg"] = - 170

    for tag in tags:
        # print(f"Tag: {tag} Min:{min(tags[tag])} Max:{max(tags[tag])} Count:{len(tags[tag])} Avg:{round(sum(tags[tag])/len(tags[tag]),2)}")

        return_dict[tag][dist][pwr][iter]["avg_time_dif"] = dif_sum / cont
        return_dict[tag][dist][pwr][iter]["rssi_min"] = min(tags[tag])
        return_dict[tag][dist][pwr][iter]["rssi_max"] = max(tags[tag])
        return_dict[tag][dist][pwr][iter]["rssi_avg"] = - round(sum(tags[tag])/len(tags[tag]),2)
        return_dict[tag][dist][pwr][iter]["activations"] = len(tags[tag]) 

        # if tag in return_dict:
        #     if dist in return_dict[tag]:
        #         if pwr in return_dict[tag][dist]:
        #             if iter in return_dict[tag][dist][pwr]:        
        #                 return_dict[tag][dist][pwr][iter]["rssi_min"] = min(tags[tag])
        #                 return_dict[tag][dist][pwr][iter]["rssi_min"] = max(tags[tag])
        #                 return_dict[tag][dist][pwr][iter]["rssi_avg"] = round(sum(tags[tag])/len(tags[tag]),2)
        #                 return_dict[tag][dist][pwr][iter]["activations"] = len(tags[tag])
        # else:
        #     return_dict[tag] = {}
    
# print(return_dict)
# print("Return: "+str(dict(return_dict["e28068940000400386621d4f"]["1.5"]["290"])))
dist_ar = []
activations_ar = []
cont = 0
data_to_plot = {"distance": []}
data_to_csv = {"distance": []}
train_dict = {"distance": []}
test_dict = {"distance": []}
for tag in return_dict:
    dist = 0.5
    while dist <= 4:
        dist_ar.append(dist)
        test_condition = True # dist >= 3 # To test with several distances intervals
        dist = str(dist)
        # print(f'Distance: {dist}')
        smpl_pwr_ar = []
        smpl_actv_ar = []
        time_dif_ar = []
        rssi_ar = []
        for pwr in range(100, 310, 10):
        # for pwr in [280, 290, 300]:
            smpl_pwr_ar.append(pwr)
            pwr = str(pwr)
            iter_avg = [] # used to make the avg and then plot
            iter_time_dif_avg = []
            iter_rssi_avg = []
            for iter in return_dict[tag][dist][pwr]:
                val = return_dict[tag][dist][pwr][iter]["activations"]
                avg_time_dif = return_dict[tag][dist][pwr][iter]["avg_time_dif"]
                iter_avg.append(val)
                iter_time_dif_avg.append(avg_time_dif)
                iter_rssi_avg.append(return_dict[tag][dist][pwr][iter]["rssi_avg"])
                cont += 1
                if int(iter) >= 3 :
                    if test_condition:
                        if pwr in test_dict:
                            test_dict[pwr].append(val)
                            test_dict[f'avg_time_dif_{pwr}'].append(avg_time_dif)
                        else:
                            test_dict[pwr] = [val]
                            test_dict[f'avg_time_dif_{pwr}'] = [avg_time_dif]
                else:
                    if pwr in train_dict:
                        train_dict[pwr].append(val)
                        train_dict[f'avg_time_dif_{pwr}'].append(avg_time_dif)
                    else:
                        train_dict[pwr] = [val]
                        train_dict[f'avg_time_dif_{pwr}'] = [avg_time_dif]

                if pwr in data_to_csv:
                    data_to_csv[pwr].append(val)
                    data_to_csv[f'avg_time_dif_{pwr}'].append(avg_time_dif)
                else:
                    data_to_csv[pwr] = [val]
                    data_to_csv[f'avg_time_dif_{pwr}'] = [avg_time_dif]
                
                # Append Y value just once per line (last pwr value)
                if pwr == "300":
                    if int(iter) >= 3 :
                        if test_condition:
                            test_dict["distance"].append(float(dist))
                    else:
                        train_dict["distance"].append(float(dist))
                    data_to_csv["distance"].append(float(dist))

            # Code used to plot the data
            if pwr != "260" and pwr != "270":
                if len(iter_avg) > 0:
                    avg = round(sum(iter_avg)/len(iter_avg),2)
                    time_dif_avg = round(sum(iter_time_dif_avg)/len(iter_time_dif_avg),2)
                    rssi_avg = round(sum(iter_rssi_avg)/len(iter_rssi_avg),2)
                else:
                    avg = iter_avg
                    time_dif_avg = iter_time_dif_avg
                    rssi_avg = iter_rssi_avg
                activations_ar.append(avg)
                smpl_actv_ar.append(avg)
                time_dif_ar.append(time_dif_avg)
                rssi_ar.append(rssi_avg)
                if pwr in data_to_plot:
                    data_to_plot[pwr].append(avg)
                else:
                    data_to_plot[pwr] = [avg]
            else:
                activations_ar.append(activations_ar[-1])
                smpl_actv_ar.append(smpl_actv_ar[-1])
                time_dif_ar.append(time_dif_ar[-1])
                rssi_ar.append(rssi_ar[-1])
        
        data_to_plot["distance"].append(float(dist))
        plt.plot(smpl_pwr_ar, smpl_actv_ar, label = f'{dist}m')
        # plt.plot(smpl_pwr_ar, time_dif_ar, label = f'{dist}m')
        # plt.plot(smpl_pwr_ar, rssi_ar, label = f'{dist}m')

        plt.xticks(smpl_pwr_ar, smpl_pwr_ar)
        # plt.show()

        dist = float(dist) + 0.5

    plt.title(tag)
    plt.xlabel('Power')
    plt.ylabel('Activations')
    # plt.ylabel('Average Time difference')
    # plt.ylabel('Average RSSI')
    plt.legend()
    # plt.show()

# print(return_dict["e28068940000400386621d4f"]["1.5"]["110"]["3"]["activations"])
# print(cont)
# print(dist_ar)
# print(len(activations_ar))

# for key in test_dict.keys():
#     print(f'{key} - {len(test_dict[key])}')

df = pd.DataFrame(data_to_plot)
train_df = pd.DataFrame(train_dict)
test_df = pd.DataFrame(test_dict)
csv_df = pd.DataFrame(data_to_csv)
# print(train_df)
# print(test_df)

# print(data_to_csv)

# # Save to CSV
# with open('antenna_dataset.csv', 'w') as f:
#     w = csv.DictWriter(f, list(data_to_csv.keys()))
#     w.writeheader()
#     for i in range(len(data_to_csv["300"])):
#         dict = {}
#         for key in data_to_csv.keys():
#             dict[key] = data_to_csv[key][i]
#         w.writerow(dict)

# print(list(data_to_csv.keys()))
# csv_file = "antenna_dataset.csv"
# try:
#     with open(csv_file, 'w') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=list(data_to_csv.keys()))
#         writer.writeheader()
#         for data in data_to_csv:
#             writer.writerow(data)
# except IOError:
#     print("I/O error")

# pd.DataFrame.from_dict(data=data_to_csv, orient='index').to_csv('antenna_dataset.csv', header=True)

x_train = train_df.drop(["distance"], axis=1)
x_test = test_df.drop(["distance"], axis=1)
y_train = train_df[["distance"]]
y_test = test_df[["distance"]]

# Feature drops
# for pwr in range(100, 310, 10):
#     # x_train = x_train.drop([str(pwr)], axis=1)
#     # x_test = x_test.drop([str(pwr)], axis=1) 
#     x_train = x_train.drop([f'avg_time_dif_{pwr}'], axis=1)
#     x_test = x_test.drop([f'avg_time_dif_{pwr}'], axis=1)

# Scale the training data
# scaler = StandardScaler()
scaler = MinMaxScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train))
x_test = pd.DataFrame(scaler.transform(x_test))

# # save the scaler
# pickle.dump(scaler, open('scaler.pkl', 'wb'))

def train_models(x_train, x_test, y_train, y_test):
    print(y_test.shape[0])
    # Training model and checking the best K
    rmse_val = [] #to store rmse values for different k
    for K in range(y_test.shape[0]):
        K = K+1
        model = neighbors.KNeighborsRegressor(n_neighbors = K)
        # model = neighbors.KNeighborsRegressor(n_neighbors = K, metric = "manhattan")
        model.fit(x_train.values, y_train.values.ravel())  #fit the model
        pred=model.predict(x_test.values) #make prediction on test set
        error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
        rmse_val.append(error) #store rmse values
        print('RMSE value for k= ' , K , 'is:', error)

    # Training model with the best K
    k_min = 1 + min(range(len(rmse_val)), key=rmse_val.__getitem__)
    model3 = neighbors.KNeighborsRegressor(n_neighbors = k_min)
    # model = neighbors.KNeighborsRegressor(n_neighbors = k_min, metric = "manhattan")
    model3.fit(x_train.values, y_train.values.ravel())  #fit the model
    pred3=model3.predict(x_test.values) #make prediction on test set
    error3 = sqrt(mean_squared_error(y_test,pred3)) #calculate rmse
    error_per3 = (error3/4.75)*100

    model1 = RandomForestRegressor() 
    model1.fit(x_train.values, y_train.values.ravel())
    pred1=model1.predict(x_test.values) #make prediction on test set
    error1 = sqrt(mean_squared_error(y_test,pred1)) #calculate rmse
    error_per1 = (error1/4.75)*100

    model2 = DecisionTreeRegressor(random_state = 0) 
    model2.fit(x_train.values, y_train.values.ravel())
    pred2=model2.predict(x_test.values) #make prediction on test set
    error2 = sqrt(mean_squared_error(y_test,pred2)) #calculate rmse
    error_per2 = (error2/4.75)*100

    model4 = GradientBoostingRegressor() 
    model4.fit(x_train.values, y_train.values.ravel())
    pred4=model4.predict(x_test.values) #make prediction on test set
    error4 = sqrt(mean_squared_error(y_test,pred4)) #calculate rmse
    error_per4 = (error4/4.75)*100

    model5 = SVR(kernel = 'rbf')
    model5.fit(x_train.values, y_train.values.ravel())
    pred5=model5.predict(x_test.values) #make prediction on test set
    error5 = sqrt(mean_squared_error(y_test,pred5)) #calculate rmse
    error_per5 = (error5/4.75)*100

    print("Errors:")
    print("Random Forest: "+"{:.2f}".format(error1)+"m ("+"{:.1f}".format(error_per1)+" %)")
    print("Decision Tree: "+"{:.2f}".format(error2)+"m ("+"{:.1f}".format(error_per2)+" %)")
    print("Support Vector Regression: "+"{:.2f}".format(error5)+"m ("+"{:.1f}".format(error_per5)+" %)")
    print("Gradient Boosting Regression: "+"{:.2f}".format(error4)+"m ("+"{:.1f}".format(error_per4)+" %)")
    print("KNN with k = "+str(k_min)+": "+"{:.2f}".format(error3)+"m ("+"{:.1f}".format(error_per3)+" %)")

    # save the model to disk
    pickle.dump(model1, open(f'random_forest_model_{x_train.shape[1]}_features.sav', 'wb'))
    pickle.dump(model2, open(f'decision_tree_model_{x_train.shape[1]}_features.sav', 'wb'))
    pickle.dump(model3, open(f'knn_model_{x_train.shape[1]}_features.sav', 'wb'))
    pickle.dump(model4, open(f'grad_boost_regr_model_{x_train.shape[1]}_features.sav', 'wb'))
    pickle.dump(model5, open(f'svr_model_{x_train.shape[1]}_features.sav', 'wb'))

    return min([error1, error2, error3, error4, error5])

# # Iterations Analysis
# print("\nIterations Analysis:")
# total_max_amp_ar = []
# total_max_amp_per_ar = []
# dist = 0.25
# while dist <= 5:
#     for pwr in range(280, 310, 10):
#         iter_ar = []
#         for iter in return_dict[tag][str(dist)][str(pwr)]:
#             iter_ar.append(return_dict["e28068940000400386621d4f"][str(dist)][str(pwr)][iter]["activations"])
#         max_amp = max(iter_ar)-min(iter_ar)
#         total_max_amp_ar.append(max_amp)
#         if max_amp == 0:
#             max_amp_per = 0
#             total_max_amp_per_ar.append(0)
#         else:
#             max_amp_per = (max_amp/max(iter_ar))*100
#             total_max_amp_per_ar.append(max_amp_per)
#             print(f'{dist} | {pwr} - {max_amp} ({max_amp_per} %)')
#     dist = dist + 0.25

# print("Average maximum amplitude: "+"{:.1f}".format(sum(total_max_amp_ar)/len(total_max_amp_ar)))
# print("Average maximum amplitude percentage: "+"{:.1f}".format(sum(total_max_amp_per_ar)/len(total_max_amp_per_ar))+" %")


# ML Models Analysis
print("\nML Models Analysis:")
# print("\nWith the whole data:")
print(f'x_train shape: {x_train.shape}')
train_models(x_train, x_test, y_train, y_test)

# k_best = SelectKBest(mutual_info_regression, k=3)
# x_train_3 = pd.DataFrame(k_best.fit_transform(x_train, y_train.values.ravel()))
# x_test_3 = pd.DataFrame(k_best.transform(x_test))
# # print(x_train)
# print("\nFeatures Selected by the mutual_info_regression algorithm: ")
# print(k_best.get_feature_names_out())

# print("\nWith just 3 features:")
# print(f'x_train shape: {x_train_3.shape}')
# train_models(x_train_3, x_test_3, y_train, y_test)


# # Cálculo da melhor combinação de 3 potências
# min_error = 999
# min_error_pwr = ""
# for pwr1 in range(200, 290, 10): # until 300 - 20
#     pwr2 = pwr1 + 10
#     while pwr2 <= 290:
#         pwr3 = pwr2 + 10
#         while pwr3 <= 300:
#             print(f'\n{pwr1} - {pwr2} - {pwr3}')
#             error = train_models(x_train[[str(pwr1), str(pwr2), str(pwr3)]], x_test[[str(pwr1), str(pwr2), str(pwr3)]], y_train, y_test)
#             print(error)
#             if error < min_error:
#                 min_error = error
#                 min_error_pwr = f'{pwr1} - {pwr2} - {pwr3}'
#             pwr3 += 10
#         pwr2 += 10
# print(min_error_pwr)
# print(min_error)


# # Cálculo da melhor combinação de 2 potências
# min_error = 999
# min_error_pwr = ""
# for pwr1 in range(200, 300, 10): # until 300 - 10
#     pwr2 = pwr1 + 10
#     while pwr2 <= 300:
#         print(f'\n{pwr1} - {pwr2}')
#         error = train_models(x_train[[str(pwr1), str(pwr2)]], x_test[[str(pwr1), str(pwr2)]], y_train, y_test)
#         print(error)
#         if error < min_error:
#             min_error = error
#             min_error_pwr = f'{pwr1} - {pwr2}'
#         pwr2 += 10
# print(min_error_pwr)
# print(min_error)


# k_best = SelectKBest(f_regression, k=3)
# x_train_3 = pd.DataFrame(k_best.fit_transform(x_train, y_train.values.ravel()))
# x_test_3 = pd.DataFrame(k_best.transform(x_test))
# # print(x_train)
# print("\nFeatures Selected by the f_regression algorithm: ")
# print(k_best.get_feature_names_out())

# print("\nWith just 3 features:")
# print(f'x_train shape: {x_train_3.shape}')
# train_models(x_train_3, x_test_3, y_train, y_test)