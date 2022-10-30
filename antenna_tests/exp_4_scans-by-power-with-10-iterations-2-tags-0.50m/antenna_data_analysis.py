from dis import dis
from operator import indexOf
import string
import sys
import glob
from tokenize import String
from collections import defaultdict
import matplotlib.pyplot as plt
import csv
import statistics
import pandas as pd
import numpy as np
import scipy.stats as stats
import pickle
from sklearn import neighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error 
from math import sqrt
from sklearn.feature_selection import SelectKBest, f_regression, r_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler
# 1648734594.331922, sdrt-health/mac:02010b41cdec/antenna/tag/e28068940000400386621d4f, 82

max_time_diff = {"200": 0, "280": 0, "290": 0, "300": 0}

def nested_dict(n, type):
    if n == 1:
        return defaultdict(type)
    else:
        return defaultdict(lambda: nested_dict(n-1, type))

def pre_process_timestamps(lst, hop = 1):
    # pre process the timestamps
    return_lst = []
    i = 0
    while i + hop < len(lst):
        # return_lst.append((lst[i+hop]+lst[i])/2) # This one has a better performance - maybe because of the passing of time as the distance is increasing
        return_lst.append((lst[i+hop]-lst[i])/hop)
        i += 1
    # if not return_lst:
    #     return_lst = [0]
    return return_lst

def pre_process_time_dif(lst, max_time):
    # pre process time differences 
    # identify outliers in the training dataset

    # try:
    #     # oi = IsolationForest(contamination=0.1)
    #     # oi = EllipticEnvelope(contamination=0.1)
    #     # oi = LocalOutlierFactor()
    #     oi = OneClassSVM()
    #     yhat = oi.fit_predict(pd.DataFrame(lst))
    #     # select all rows that are not outliers
    #     mask = yhat != -1
    #     lst = pd.DataFrame(lst)[mask].values.reshape(-1).tolist()
    # except:
    #     print("Exception  caught") # When there is just 1 or 0 elements on the list -> pass

    # df = pd.DataFrame(lst)
    # #find absolute value of z-score for each observation
    # z = np.abs(stats.zscore(df))
    # #only keep rows in dataframe with all z-scores less than absolute value of 3 
    # lst = df[(z<3).all(axis=1)].values.reshape(-1).tolist()

    # df = pd.DataFrame(lst)
    # # find Q1, Q3, and interquartile range for each column
    # Q1 = df.quantile(q=.25)
    # Q3 = df.quantile(q=.75)
    # IQR = df.apply(stats.iqr)
    # #only keep rows in dataframe that have values within 1.5*IQR of Q1 and Q3
    # lst = (df[~((df < (Q1-1.5*IQR)) | (df > (Q3+1.5*IQR))).any(axis=1)]).values.reshape(-1).tolist()

    if not lst:
        lst = [max_time]
    
    return lst

tag_ar = ["e28068940000400386621d4f", "e2806894000050038661fd4f", "e2806894000050038662154f", "e2806894000050038662194f",\
        "e2806894000040038662114f", "e2806894000040038662054f", "e2806894000040038661f94f", "e2806894000040038662214f",\
        "e2806894000050038662014f", "e28068940000500386620d4f"]

DYNAMIC_TAG = ["e28068940000400386621d4f", "e2806894000040038662214f"]

data_dir_name = ""
return_dict = {}
return_dict = nested_dict(5, float)
data_files = list(i.split("/")[-1] for i in glob.glob(data_dir_name+"*/*"))
data_files = sorted(data_files, key = lambda x: float(x.split("_")[1]))
# print("")
it = 0
for data_file in data_files:
    file_lst = data_file.replace(".csv", "").split("_")
    dist = file_lst[1]
    pwr = file_lst[3]
    iter = file_lst[-1]
    # print(dist, pwr, iter)

    f = open(f'{data_dir_name}{dist}_meters/{data_file}', 'r')

    tags_rssi = dict()
    ver = False
     
    cont = {"e28068940000400386621d4f": 0, "e2806894000050038661fd4f": 0, "e2806894000050038662154f": 0, "e2806894000050038662194f": 0,\
        "e2806894000040038662114f": 0, "e2806894000040038662054f": 0, "e2806894000040038661f94f": 0, "e2806894000040038662214f": 0,\
        "e2806894000050038662014f": 0, "e28068940000500386620d4f": 0,}
    prev_timestamp = {"e28068940000400386621d4f": 0, "e2806894000050038661fd4f": 0, "e2806894000050038662154f": 0, "e2806894000050038662194f": 0,\
        "e2806894000040038662114f": 0, "e2806894000040038662054f": 0, "e2806894000040038661f94f": 0, "e2806894000040038662214f": 0,\
        "e2806894000050038662014f": 0, "e28068940000500386620d4f": 0,}
    dif_lst = {"e28068940000400386621d4f": [], "e2806894000050038661fd4f": [], "e2806894000050038662154f": [], "e2806894000050038662194f": [],\
        "e2806894000040038662114f": [], "e2806894000040038662054f": [], "e2806894000040038661f94f": [], "e2806894000040038662214f": [],\
        "e2806894000050038662014f": [], "e28068940000500386620d4f": [],}
    times_ar = {"e28068940000400386621d4f": [], "e2806894000050038661fd4f": [], "e2806894000050038662154f": [], "e2806894000050038662194f": [],\
        "e2806894000040038662114f": [], "e2806894000040038662054f": [], "e2806894000040038661f94f": [], "e2806894000040038662214f": [],\
        "e2806894000050038662014f": [], "e28068940000500386620d4f": [],}

    for line in f:
        tag = ""
        if 'antenna/tag/e2' in line:
            
            vl = line.split(", ")
            
            if(len(vl) == 3):

                ver = True
                
                tag = vl[1].split('/')[-1]
                rssi = int(vl[2])

                timestamp = float(vl[0])
                times_ar[tag].append(timestamp)
                dif_lst[tag].append(timestamp - prev_timestamp[tag])
                prev_timestamp[tag] = timestamp
                cont[tag]+=1

                if tag not in tags_rssi:
                    tags_rssi[tag] = []

                tags_rssi[tag].append(rssi)

        

    for tag in tag_ar: 
        it += 1
        # print(it, "/ 5230")
        
        dif_lst[tag] = pre_process_timestamps(times_ar[tag]) # Change the calculus of the difference between activations
        # dif_lst[tag] = dif_lst[tag][1:]
        #    
        # if tag in DYNAMIC_TAG:
            # print(dist, pwr, iter, len(dif_lst[tag]))
        
        if not tag in tags_rssi:
            return_dict[tag][dist][pwr][iter]["avg_time_dif"] = max_time_diff[pwr]
            return_dict[tag][dist][pwr][iter]["activations"] = 0
            return_dict[tag][dist][pwr][iter]["rssi_min"] = 170
            return_dict[tag][dist][pwr][iter]["rssi_max"] = 170
            return_dict[tag][dist][pwr][iter]["rssi_avg"] = 170

        else:
            dif_lst[tag] = pre_process_time_dif(dif_lst[tag], max_time_diff[pwr])

            if tag in DYNAMIC_TAG:
                mean = statistics.mean(dif_lst[tag])
                if mean > max_time_diff[pwr]:
                    # print("\n", dist, pwr, iter, max_time_diff[pwr])
                    max_time_diff[pwr] = mean
                    # print(max_time_diff[pwr])

            return_dict[tag][dist][pwr][iter]["avg_time_dif"] = statistics.mean(dif_lst[tag])
            # return_dict[tag][dist][pwr][iter]["avg_time_dif"] = statistics.median(dif_lst[tag])
            return_dict[tag][dist][pwr][iter]["rssi_min"] = min(tags_rssi[tag])
            return_dict[tag][dist][pwr][iter]["rssi_max"] = max(tags_rssi[tag])
            return_dict[tag][dist][pwr][iter]["rssi_avg"] = round(sum(tags_rssi[tag])/len(tags_rssi[tag]),2)
            return_dict[tag][dist][pwr][iter]["activations"] = len(tags_rssi[tag])

        # if tag in return_dict:
        #     if dist in return_dict[tag]:
        #         if pwr in return_dict[tag][dist]:
        #             if iter in return_dict[tag][dist][pwr]:        
        #                 return_dict[tag][dist][pwr][iter]["rssi_min"] = min(tags_rssi[tag])
        #                 return_dict[tag][dist][pwr][iter]["rssi_min"] = max(tags_rssi[tag])
        #                 return_dict[tag][dist][pwr][iter]["rssi_avg"] = round(sum(tags_rssi[tag])/len(tags_rssi[tag]),2)
        #                 return_dict[tag][dist][pwr][iter]["activations"] = len(tags_rssi[tag])
        # else:
        #     return_dict[tag] = {}
    


# print(return_dict)
# print("Return: "+str(dict(return_dict["e28068940000400386621d4f"]["1.5"]["290"])))
activations_ar = []
cont = 0
data_to_csv = {"distance": []}
train_dict = {"distance": []}
test_dict = {"distance": []}
test_df_ar = {}
# for tag in return_dict:
for tag in DYNAMIC_TAG: # iterate over the dynamic tags
    print(f'\nTag {tag}')
    train_dict = {"distance": []} # Comentar para gerar modelos baseados em ambas as tags
    test_dict = {"distance": []}
    dist = 0.5
    dist_ar = []
    plot_dict = {"smpl_actv": [[], [], [], []], "time_dif": [[], [], [], []], "activ_dif": [[], [], [], []]} # Contains the data per power
    # append list for each reference tag
    for pwr in range(len(plot_dict["activ_dif"])):
        for i in range(len(tag_ar)-1):
            plot_dict["activ_dif"][pwr].append([])
    while dist <= 4.0:
        dist_ar.append(dist)
        test_condition = True # dist >= 3
        dist = str(dist)
        # print(f'Distance: {dist}')
        smpl_pwr_ar = []
        smpl_actv_ar = []
        time_dif_ar = []
        activ_dif_ar = [[],[],[],[],[],[],[],[],[]]
        # for pwr in range(100, 310, 10):
        for index, pwr in enumerate([200, 280, 290, 300]):
            smpl_pwr_ar.append(pwr)
            pwr = str(pwr)
            # Average of data over the iterations to plot
            iter_avg = []
            iter_time_dif_avg = []
            iter_activ_dif_avg = [[],[],[],[],[],[],[],[],[]]
            for iter in return_dict[tag][dist][pwr]:
                val = return_dict[tag][dist][pwr][iter]["activations"]
                avg_time_dif = return_dict[tag][dist][pwr][iter]["avg_time_dif"]
                activ_dif = []
                for i in range(len(tag_ar)-1):
                    ad = return_dict[tag][dist][pwr][iter]["activations"] - return_dict[tag_ar[i]][dist][pwr][iter]["activations"]
                    activ_dif.append(ad)
                    iter_activ_dif_avg[i].append(ad)
                iter_avg.append(val)
                iter_time_dif_avg.append(avg_time_dif)
                cont += 1
                if int(iter) >= 8:
                    if test_condition:
                        if pwr in test_dict:
                            test_dict[pwr].append(val)
                            test_dict[f'avg_time_dif_{pwr}'].append(avg_time_dif)
                            for i in range(len(tag_ar)-1):
                                test_dict[f'activ_dif_{i}_{pwr}'].append(activ_dif[i])
                        else:
                            test_dict[pwr] = [val]
                            test_dict[f'avg_time_dif_{pwr}'] = [avg_time_dif]
                            for i in range(len(tag_ar)-1):
                                test_dict[f'activ_dif_{i}_{pwr}'] = [activ_dif[i]]
                else:
                    if pwr in train_dict:
                        train_dict[pwr].append(val)
                        train_dict[f'avg_time_dif_{pwr}'].append(avg_time_dif)
                        for i in range(len(tag_ar)-1):
                            train_dict[f'activ_dif_{i}_{pwr}'].append(activ_dif[i])
                    else:
                        train_dict[pwr] = [val]
                        train_dict[f'avg_time_dif_{pwr}'] = [avg_time_dif]
                        for i in range(len(tag_ar)-1):
                            train_dict[f'activ_dif_{i}_{pwr}'] = [activ_dif[i]]

                if pwr in data_to_csv:
                    data_to_csv[pwr].append(val)
                    data_to_csv[f'avg_time_dif_{pwr}'].append(avg_time_dif)
                    for i in range(len(tag_ar)-1):
                        data_to_csv[f'activ_dif_{i}_{pwr}'].append(activ_dif[i])
                else:
                    data_to_csv[pwr] = [val]
                    data_to_csv[f'avg_time_dif_{pwr}'] = [avg_time_dif]
                    for i in range(len(tag_ar)-1):
                        data_to_csv[f'activ_dif_{i}_{pwr}'] = [activ_dif[i]]
                
                # Append Y value just once per line (last pwr value)
                if pwr == "300":
                    if int(iter) >= 8:
                        if test_condition:
                            test_dict["distance"].append(float(dist))
                    else:
                        train_dict["distance"].append(float(dist))
                    data_to_csv["distance"].append(float(dist))


            if pwr != "260" and pwr != "270":
                if len(iter_avg) > 0:
                    avg = round(sum(iter_avg)/len(iter_avg),2)
                    time_dif_avg = round(sum(iter_time_dif_avg)/len(iter_time_dif_avg),2)
                    activ_dif_avg = []
                    for i in range(len(tag_ar)-1):
                        activ_dif_avg.append(round(sum(iter_activ_dif_avg[i])/len(iter_activ_dif_avg[i]),2))
                else:
                    avg = iter_avg
                    time_dif_avg = iter_time_dif_avg
                    for i in range(len(tag_ar)-1):
                        activ_dif_avg.append(iter_activ_dif_avg[i])
                activations_ar.append(avg)
                smpl_actv_ar.append(avg)
                time_dif_ar.append(time_dif_avg)
                for i in range(len(tag_ar)-1):
                    activ_dif_ar[i].append(activ_dif_avg[i])
            else:
                activations_ar.append(activations_ar[-1])
                smpl_actv_ar.append(smpl_actv_ar[-1])
                time_dif_ar.append(time_dif_ar[-1])

            plot_dict["smpl_actv"][index].append(smpl_actv_ar[-1])
            plot_dict["time_dif"][index].append(time_dif_ar[-1])

            for i in range(len(tag_ar)-1):
                plot_dict["activ_dif"][index][i].append(activ_dif_ar[i][-1])
                
        # data_to_csv["distance"].append(float(dist))

        # Plot data per power with multiple distance lines
        # Choose
        # plt.plot(smpl_pwr_ar, smpl_actv_ar, label = f'{dist}m')
        # plt.plot(smpl_pwr_ar, time_dif_ar, label = f'{dist}m')
        # plt.plot(smpl_pwr_ar, activ_dif_ar[-1], label = f'{dist}m')

        # plt.show()
        dist = float(dist) + 0.5

    # Plot data per distance with 3 power lines INV
    for index, pwr in enumerate(smpl_pwr_ar):
        # plt.plot(dist_ar, plot_dict["smpl_actv"][index], label = f'{pwr}mW')
        plt.plot(dist_ar, plot_dict["time_dif"][index], label = f'{pwr}mW')

    # for i in range(len(tag_ar)-1):
    #     plt.plot(dist_ar, plot_dict["activ_dif"][2][i], label = f'...{tag_ar[i][-5:]}')
    # plt.ylabel(f'Activations differences')
    # plt.title("300 dBm")

    # Choose
    # plt.ylabel('Activations')
    plt.ylabel('Average Time difference')

    # plt.xticks(smpl_pwr_ar, smpl_pwr_ar)
    plt.xticks(dist_ar, dist_ar) # INV

    # plt.yscale('log')
    plt.title(tag)
    # plt.xlabel('Power')
    plt.xlabel('Distance')
    plt.legend()
    # plt.show()


    # Save to CSV
    # with open(f'{data_dir_name}_dataset.csv', 'w') as f:
    #     w = csv.DictWriter(f, list(data_to_csv.keys()))
    #     w.writeheader()
    #     for i in range(len(data_to_csv["300"])):
    #         dict = {}
    #         for key in data_to_csv.keys():
    #             dict[key] = data_to_csv[key][i]
    #         w.writerow(dict)


    df = pd.DataFrame(data_to_csv)
    train_df = pd.DataFrame(train_dict)
    test_df = pd.DataFrame(test_dict)

    x_train = train_df.drop(["distance"], axis=1)
    x_test = test_df.drop(["distance"], axis=1)
    y_train = train_df[["distance"]]
    y_test = test_df[["distance"]]

    # Feature drops
    for pwr in [200, 280, 290, 300]: 
        # x_train = x_train.drop([str(pwr)], axis=1)
        # x_test = x_test.drop([str(pwr)], axis=1) 
        x_train = x_train.drop([f'avg_time_dif_{pwr}'], axis=1)
        x_test = x_test.drop([f'avg_time_dif_{pwr}'], axis=1) 
        for i in range(len(tag_ar)-1):
            x_train = x_train.drop([f'activ_dif_{i}_{pwr}'], axis=1)
            x_test = x_test.drop([f'activ_dif_{i}_{pwr}'], axis=1)

    # x_train = x_train.drop([str(300)], axis=1)
    # x_test = x_test.drop([str(300)], axis=1) 

    # # Scale the training data
    # scaler = pickle.load(open('scaler.pkl', 'rb'))
    # x_test = pd.DataFrame(scaler.transform(x_test))

    # scaler = StandardScaler()
    # x_test = pd.DataFrame(scaler.fit_transform(x_test))

    def test_model(x_test, y_test):
        # x_test = x_test.drop(["200"], axis=1)
        # x_test = x_test.drop(["avg_time_dif_200"], axis=1)

        for model_file_name in ["random_forest_model_4_features.sav", "decision_tree_model_4_features.sav", "knn_model_4_features.sav", "grad_boost_regr_model_4_features.sav", "svr_model_4_features.sav"]:
        # for model_file_name in ["../random_forest_model_6_features.sav", "../decision_tree_model_6_features.sav", "../knn_model_6_features.sav", "../grad_boost_regr_model_6_features.sav", "../svr_model_6_features.sav"]:
            loaded_model = pickle.load(open(model_file_name, 'rb'))
            pred=loaded_model.predict(x_test.values) #make prediction on test set
            error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
            error_per = (error/3.75)*100
            print(model_file_name+" {:.2f}".format(error)+"m ("+"{:.1f}".format(error_per)+" %)")

    # test_model(x_test, y_test)

    def train_models(x_train, x_test, y_train, y_test):
        # Training model and checking the best K
        dist = 3.5
        rmse_val = [] #to store rmse values for different k
        for K in range(y_test.shape[0]):
            K = K+1
            model = neighbors.KNeighborsRegressor(n_neighbors = K)
            # model = neighbors.KNeighborsRegressor(n_neighbors = K, metric = "manhattan")
            model.fit(x_train.values, y_train.values.ravel())  #fit the model
            pred=model.predict(x_test.values) #make prediction on test set
            error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
            rmse_val.append(error) #store rmse values
            # print('RMSE value for k= ' , K , 'is:', error)

        # Training model with the best K
        k_min = 1 + min(range(len(rmse_val)), key=rmse_val.__getitem__)
        model3 = neighbors.KNeighborsRegressor(n_neighbors = k_min)
        # model = neighbors.KNeighborsRegressor(n_neighbors = k_min, metric = "manhattan")
        model3.fit(x_train.values, y_train.values.ravel())  #fit the model
        pred3=model3.predict(x_test.values) #make prediction on test set
        error3 = sqrt(mean_squared_error(y_test,pred3)) #calculate rmse
        error_per3 = (error3/dist)*100

        model1 = RandomForestRegressor() 
        model1.fit(x_train.values, y_train.values.ravel())
        pred1=model1.predict(x_test.values) #make prediction on test set
        error1 = sqrt(mean_squared_error(y_test,pred1)) #calculate rmse
        error_per1 = (error1/dist)*100

        model2 = DecisionTreeRegressor(random_state = 0) 
        model2.fit(x_train.values, y_train.values.ravel())
        pred2=model2.predict(x_test.values) #make prediction on test set
        error2 = sqrt(mean_squared_error(y_test,pred2)) #calculate rmse
        error_per2 = (error2/dist)*100

        model4 = GradientBoostingRegressor() 
        model4.fit(x_train.values, y_train.values.ravel())
        pred4=model4.predict(x_test.values) #make prediction on test set
        error4 = sqrt(mean_squared_error(y_test,pred4)) #calculate rmse
        error_per4 = (error4/dist)*100

        model5 = SVR(kernel = 'rbf')
        model5.fit(x_train.values, y_train.values.ravel())
        pred5=model5.predict(x_test.values) #make prediction on test set
        error5 = sqrt(mean_squared_error(y_test,pred5)) #calculate rmse
        error_per5 = (error5/dist)*100

        print("Errors:")
        print("Random Forest: "+"{:.2f}".format(error1)+"m ("+"{:.1f}".format(error_per1)+" %)")
        print("Decision Tree: "+"{:.2f}".format(error2)+"m ("+"{:.1f}".format(error_per2)+" %)")
        print("Support Vector Regression: "+"{:.2f}".format(error5)+"m ("+"{:.1f}".format(error_per5)+" %)")
        print("Gradient Boosting Regression: "+"{:.2f}".format(error4)+"m ("+"{:.1f}".format(error_per4)+" %)")
        print("KNN with k = "+str(k_min)+": "+"{:.2f}".format(error3)+"m ("+"{:.1f}".format(error_per3)+" %)")

        return min([error1, error2, error3, error4, error5])



    # # Iterations Analysis - Differences in the activations data between iterations
    # print("\nIterations Analysis:")
    # total_max_amp_ar = []
    # total_max_amp_per_ar = []
    # dist = 0.25
    # while dist <= 5:
    #     for pwr in range(200, 280, 310, 10):
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

    # print("\nWith just 1 features:")
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


    # # Cálculo da melhor combinação de 3 potências
    # k_best = SelectKBest(f_regression, k=3)
    # x_train_3 = pd.DataFrame(k_best.fit_transform(x_train, y_train.values.ravel()))
    # x_test_3 = pd.DataFrame(k_best.transform(x_test))
    # # print(x_train)
    # print("\nFeatures Selected by the f_regression algorithm: ")
    # print(k_best.get_feature_names_out())

    # print("\nWith just 3 features:")
    # print(f'x_train shape: {x_train_3.shape}')
    # train_models(x_train_3, x_test_3, y_train, y_test)

    # print(pre_process_timestamps([1, 2, 3, 4, 5]))