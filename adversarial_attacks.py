from sklearn.utils import class_weight
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_blobs
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, cross_val_score
import skorch
from skorch import NeuralNetClassifier
import os
from sklearn.utils.class_weight import compute_class_weight
import shutil
import warnings
from pyds import MassFunction
from tabulate import tabulate
import random

# import foolbox as fb
# import torchattacks

# from google.colab import files
import warnings


from art.attacks.evasion import DeepFool, SaliencyMapMethod, ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"

    df = df.replace('NaN', np.nan)
    df = df.replace('Infinity', np.nan)
    df = df.replace('infinity', np.nan)
    df = df.replace('inf', np.nan)
    df = df.replace(np.inf, np.nan)
    df = df.replace('', np.nan)
    df = df.dropna(axis=0)

    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)

    # return df[indices_to_keep].astype(np.float64)
    return df[indices_to_keep]

def get_data():
    '''
    1. Opens a CSV dataset file and add to a dataframe\n
    2. Variables for raw, normalised and Labels  (rX, nX, Y, respectively)\n
    3. Manually add feature names (This could be automated if CSV file has Header/titles)\n
    '''

    technique = "all_features"

    print("Pulling in the dataset... ", end="", flush=True)
    df = pd.read_csv('1502_combined_attacks.csv', low_memory=False)
    # df = pd.read_csv('1402_canadian.csv', low_memory=False)
    print("Done!")

    print("Cleaning up dataset... ", end="", flush=True)
    df = clean_dataset(df)
    print("Done!")

    data = df.values
    X = data[:, :-1]

    rX = np.copy(X)
    nX = np.copy(X)

    feature_names = list(df.columns)
    del feature_names[-1]

    y = data[:, -1]

    print("Calculating the mean values... ", end="", flush=True)
    features_mean = np.mean(rX, axis=0)
    print("Done!")

    print("Calculating the std values... ", end="", flush=True)
    features_std = np.std(rX, axis=0)
    print("Done!")

    features_std = np.where(features_std != 0, features_std, 1.0)

    print("Normalising the dataset... ", end="", flush=True)
    nX = (nX - features_mean) / features_std
    print("Done!")

    return nX, rX, y, feature_names, technique

def calculate_masses(layer2, beta, bias, y):
    mass_function_1 = []
    mass_function_2 = []
    mass_function_3 = []
    mass_function_12 = []
    mass_function_13 = []
    mass_function_23 = []
    mass_function_123 = []
    conflict_list = []
    eta_list = []
    eta_plus_list = []
    eta_neg_list = []


    mass_list = []

    K = model.output.weight.shape[0] # 3 (final layer)
    J = model.output.weight.shape[1] # 10 (second hidden layer) 

    np.set_printoptions(precision=18)

    phi = layer2.cpu().detach().numpy()
    mu = np.mean(layer2.cpu().detach().numpy(), axis=0)

    beta_star_jk = beta - (1/K) * np.sum(beta, axis=0)
    # print("beta_star_jk", beta_star_jk)

    beta_dash_0_k = bias + np.sum(beta * mu, axis=1)
    # print("beta_dash_0_k", beta_dash_0_k)

    beta_dash_star_0_k = beta_dash_0_k - (1/K) * np.sum(beta_dash_0_k)
    # print("beta_dash_star_0_k", beta_dash_star_0_k)

    beta_star_0_k = bias - (1/K) * np.sum(bias)
    # print("beta_star_0_k", beta_star_0_k)

    alpha_dash_star_jk = beta_dash_star_0_k / J
    # print("alpha_dash_star_jk", alpha_dash_star_jk)

    alpha_star_jk = alpha_dash_star_jk - (beta_star_jk * mu).T # non-linear
    # print("alpha_star_jk", alpha_star_jk)

    # alpha_star_jk = beta_star_0_k / J # linear

    for i in phi:
        wjk = (beta * i).T + alpha_star_jk
        pos_part = np.maximum(0, wjk)
        neg_part = np.maximum(0, -wjk)
        w_plus = pos_part.sum(axis=0)
        w_neg = neg_part.sum(axis=0)
        eta_plus = (np.sum(np.exp(w_plus)) - K + 1) ** (-1)
        eta_neg = (1 - np.prod(1 - np.exp(-w_neg))) ** (-1)
        k_conflict = np.sum(eta_plus * (np.exp(w_plus) - 1) * (1 - eta_neg * np.exp(-w_neg)))

        if k_conflict == 1:
            k_conflict = k_conflict - 0.000001
        
        eta = (1 - k_conflict) ** (-1)

        # print("mu", mu)
        # print("wjk", wjk)
        # print("pos_part", pos_part)
        # print("neg_part", neg_part)
        # print("w_plus", w_plus)
        # print("w_neg", w_neg)
        # print("eta_plus", eta_plus)
        # print("eta_neg", eta_neg)
        # print("k_conflict", k_conflict)
        # print("eta", eta)
        # sys.exit()

        m1 = eta * eta_neg * eta_plus * np.exp(-w_neg[0]) * ( np.exp(w_plus[0]) - 1 + ((1 - np.exp(-w_neg[1])) * (1 - np.exp(-w_neg[2]))) )
        m2 = eta * eta_neg * eta_plus * np.exp(-w_neg[1]) * ( np.exp(w_plus[1]) - 1 + ((1 - np.exp(-w_neg[0])) * (1 - np.exp(-w_neg[2]))) )
        m3 = eta * eta_neg * eta_plus * np.exp(-w_neg[2]) * ( np.exp(w_plus[2]) - 1 + ((1 - np.exp(-w_neg[0])) * (1 - np.exp(-w_neg[1]))) )
        m12 = eta * eta_neg * eta_plus * ( 1 - np.exp(-w_neg[2]) ) * ( np.exp(-w_neg[0]) * np.exp(-w_neg[1]) )
        m13 = eta * eta_neg * eta_plus * ( 1 - np.exp(-w_neg[1]) ) * ( np.exp(-w_neg[0]) * np.exp(-w_neg[2]) )
        m23 = eta * eta_neg * eta_plus * ( 1 - np.exp(-w_neg[0]) ) * ( np.exp(-w_neg[1]) * np.exp(-w_neg[2]) )
        m123 = eta * eta_neg * eta_plus * ( np.exp(-w_neg[0]) * np.exp(-w_neg[1]) * np.exp(-w_neg[2]) )

        mass_list.append([m1, m2, m3, m12, m13, m23, m123, k_conflict])

        mass_function_1.append(m1)
        mass_function_2.append(m2)
        mass_function_3.append(m3)
        mass_function_12.append(m12)
        mass_function_13.append(m13)
        mass_function_23.append(m23)
        mass_function_123.append(m123)
        conflict_list.append(k_conflict)
        eta_list.append(eta)
        eta_plus_list.append(eta_plus)
        eta_neg_list.append(eta_neg)

    mass_list = np.array(mass_list)

    return mass_list, mass_function_1, mass_function_2, mass_function_3, mass_function_12, mass_function_13, mass_function_23, mass_function_123, conflict_list, eta_list, eta_plus_list, eta_neg_list

def join_mass_functions(mass_function_1, mass_function_2, mass_function_3, mass_function_12, mass_function_13, mass_function_23, mass_function_123):
    mass_functions = []

    for t1, t2, t3, t12, t13, t23, t123 in zip(mass_function_1, mass_function_2, mass_function_3, mass_function_12, mass_function_13, mass_function_23, mass_function_123):
        mass_functions.append(MassFunction({'1': t1, '2': t2, '3': t3, '12': t12, '13': t13, '23': t23, '123': t123}))

    return mass_functions

def calculate_belief(mass_functions):
    t1_belief_list = []
    t2_belief_list = []
    t3_belief_list = []

    for i in mass_functions:
        t1_belief_list.append(i.bel('1'))
        t2_belief_list.append(i.bel('2'))
        t3_belief_list.append(i.bel('3'))

    t1_belief_array = np.asarray(t1_belief_list)
    t2_belief_array = np.asarray(t2_belief_list)
    t3_belief_array = np.asarray(t3_belief_list)

    return t1_belief_array, t2_belief_array, t3_belief_array

def calculate_plausibility(mass_functions):
    t1_plausibility_list = []
    t2_plausibility_list = []
    t3_plausibility_list = []

    for i in mass_functions:
        t1_plausibility_list.append(i.pl('1'))
        t2_plausibility_list.append(i.pl('2'))
        t3_plausibility_list.append(i.pl('3'))

    t1_plausibility_array = np.asarray(t1_plausibility_list)
    t2_plausibility_array = np.asarray(t2_plausibility_list)
    t3_plausibility_array = np.asarray(t3_plausibility_list)

    return t1_plausibility_array, t2_plausibility_array, t3_plausibility_array

def calculate_mp_rule(t1_plausibility_array, t2_plausibility_array, t3_plausibility_array):
    decision_mp = []

    for theta1, theta2, theta3 in zip(t1_plausibility_array, t2_plausibility_array, t3_plausibility_array):
        if (1 - theta1) <= (1 - theta2) and (1 - theta1) <= (1 - theta3):
            decision_mp.append(0)
        elif (1 - theta2) <= (1 - theta1) and (1 - theta2) <= (1 - theta3):
            decision_mp.append(1)
        # elif (1 - theta3) <= (1 - theta1) and (1 - theta3) <= (1 - theta2):
        else:
            decision_mp.append(2)

    return decision_mp

def calculate_ca_rule(t1_plausibility_array, t2_plausibility_array, t3_plausibility_array, t1_belief_array, t2_belief_array, t3_belief_array):
    decision_ca = []

    # R^*(A) = 1 - Bel({θk})
    # Rv*(A) = 1 - Pl({θk})

    # Rv*(A) <= R^*('A)

    for pl_theta1, pl_theta2, pl_theta3, bel_theta1, bel_theta2, bel_theta3 in zip(t1_plausibility_array, t2_plausibility_array, t3_plausibility_array, t1_belief_array, t2_belief_array, t3_belief_array):
        if (1 - bel_theta1) <= (1 - pl_theta2) and (1 - bel_theta1) <= (1 - pl_theta3):
            decision_ca.append(0)
        elif (1 - bel_theta2) <= (1 - pl_theta1) and (1 - bel_theta2) <= (1 - pl_theta3):
            decision_ca.append(1)
        # elif (1 - bel_theta3) <= (1 - pl_theta1) and (1 - bel_theta3) <= (1 - pl_theta2):
        else:
            decision_ca.append(2)

    return decision_ca

def calculate_me_rule(t1_plausibility_array, t2_plausibility_array, t3_plausibility_array, t1_belief_array, t2_belief_array, t3_belief_array):
    decision_me = []

    # R^*(A) = 1 - Bel({θk})
    # Rv*(A) = 1 - Pl({θk})

    # Rv*(A) < R^*('A)

    for pl_theta1, pl_theta2, pl_theta3, bel_theta1, bel_theta2, bel_theta3 in zip(t1_plausibility_array, t2_plausibility_array, t3_plausibility_array, t1_belief_array, t2_belief_array, t3_belief_array):
        if (1 - pl_theta1) < (1 - bel_theta2) and (1 - pl_theta1) < (1 - bel_theta3):
            decision_me.append(0)
        elif (1 - pl_theta2) < (1 - bel_theta1) and (1 - pl_theta2) < (1 - bel_theta3):
            decision_me.append(1)
        # elif (1 - pl_theta3) < (1 - bel_theta1) and (1 - pl_theta3) < (1 - bel_theta2):
        else:
            decision_me.append(2)

    return decision_me

def calculate_id_rule(decision_me, decision_ca):
    decision_id = []

    # for me, ca in zip(decision_me, decision_ca):
    #     if me == 0 and ca == 0:
    #         decision_id.append(1)
    #     elif me == 1 and ca == 1:
    #         decision_id.append(2)
    #     elif me == 2 and ca == 2:
    #         decision_id.append(3)
    #     elif (me == 0 and ca == 1) or (me == 1 and ca == 0):
    #         decision_id.append(12)
    #     elif (me == 0 and ca == 2) or (me == 2 and ca == 0):
    #         decision_id.append(13)
    #     elif (me == 1 and ca == 2) or (me == 2 and ca == 1):
    #         decision_id.append(23)
    #     else:
    #         decision_id.append(123)

    for me, ca in zip(decision_me, decision_ca):
        if me == 0 and ca == 0:
            decision_id.append(1)
        elif me == 1 and ca == 1:
            decision_id.append(2)
        elif me == 2 and ca == 2:
            decision_id.append(3)
        else:
            decision_id.append(123)

    return decision_id

def tabulate_id_rule(id_rule, gt):
    # c1p1 = []
    # c1p2 = []
    # c1p3 = []
    # c1p12 = []
    # c1p13 = []
    # c1p23 = []
    # c1p123 = []

    # c2p1 = []
    # c2p2 = []
    # c2p3 = []
    # c2p12 = []
    # c2p13 = []
    # c2p23 = []
    # c2p123 = []

    # c3p1 = []
    # c3p2 = []
    # c3p3 = []
    # c3p13 = []
    # c3p12 = []
    # c3p23 = []
    # c3p123 = []


    # for id, gt in zip(id_rule, y):
    #     if gt == 0 and id == 1:
    #         c1p1.append(1)
    #         c1p2.append(0)
    #         c1p3.append(0)
    #         c1p123.append(0)

    #         c2p1.append(0)
    #         c2p2.append(0)
    #         c2p3.append(0)
    #         c2p123.append(0)

    #         c3p1.append(0)
    #         c3p2.append(0)
    #         c3p3.append(0)
    #         c3p123.append(0)

    #     elif gt == 0 and id == 2:
    #         c1p1.append(0)
    #         c1p2.append(1)
    #         c1p3.append(0)
    #         c1p123.append(0)

    #         c2p1.append(0)
    #         c2p2.append(0)
    #         c2p3.append(0)
    #         c2p123.append(0)

    #         c3p1.append(0)
    #         c3p2.append(0)
    #         c3p3.append(0)
    #         c3p123.append(0)

    #     elif  gt == 0 and id == 3:
    #         c1p1.append(0)
    #         c1p2.append(0)
    #         c1p3.append(1)
    #         c1p123.append(0)

    #         c2p1.append(0)
    #         c2p2.append(0)
    #         c2p3.append(0)
    #         c2p123.append(0)

    #         c3p1.append(0)
    #         c3p2.append(0)
    #         c3p3.append(0)
    #         c3p123.append(0)

    #     elif gt == 0 and id == 123:
    #         c1p1.append(0)
    #         c1p2.append(0)
    #         c1p3.append(0)
    #         c1p123.append(1)

    #         c2p1.append(0)
    #         c2p2.append(0)
    #         c2p3.append(0)
    #         c2p123.append(0)

    #         c3p1.append(0)
    #         c3p2.append(0)
    #         c3p3.append(0)
    #         c3p123.append(0)

    #     elif gt == 1 and id == 1:
    #         c1p1.append(0)
    #         c1p2.append(0)
    #         c1p3.append(0)
    #         c1p123.append(0)

    #         c2p1.append(1)
    #         c2p2.append(0)
    #         c2p3.append(0)
    #         c2p123.append(0)

    #         c3p1.append(0)
    #         c3p2.append(0)
    #         c3p3.append(0)
    #         c3p123.append(0)

    #     elif gt == 1 and id == 2:
    #         c1p1.append(0)
    #         c1p2.append(0)
    #         c1p3.append(0)
    #         c1p123.append(0)

    #         c2p1.append(0)
    #         c2p2.append(1)
    #         c2p3.append(0)
    #         c2p123.append(0)

    #         c3p1.append(0)
    #         c3p2.append(0)
    #         c3p3.append(0)
    #         c3p123.append(0)

    #     elif  gt == 1 and id == 3:
    #         c1p1.append(0)
    #         c1p2.append(0)
    #         c1p3.append(0)
    #         c1p123.append(0)

    #         c2p1.append(0)
    #         c2p2.append(0)
    #         c2p3.append(1)
    #         c2p123.append(0)

    #         c3p1.append(0)
    #         c3p2.append(0)
    #         c3p3.append(0)
    #         c3p123.append(0)

    #     elif gt == 1 and id == 123:
    #         c1p1.append(0)
    #         c1p2.append(0)
    #         c1p3.append(0)
    #         c1p123.append(0)

    #         c2p1.append(0)
    #         c2p2.append(0)
    #         c2p3.append(0)
    #         c2p123.append(1)

    #         c3p1.append(0)
    #         c3p2.append(0)
    #         c3p3.append(0)
    #         c3p123.append(0)

    #     elif gt == 2 and id == 1:
    #         c1p1.append(0)
    #         c1p2.append(0)
    #         c1p3.append(0)
    #         c1p123.append(0)

    #         c2p1.append(0)
    #         c2p2.append(0)
    #         c2p3.append(0)
    #         c2p123.append(0)

    #         c3p1.append(1)
    #         c3p2.append(0)
    #         c3p3.append(0)
    #         c3p123.append(0)

    #     elif gt == 2 and id == 2:
    #         c1p1.append(0)
    #         c1p2.append(0)
    #         c1p3.append(0)
    #         c1p123.append(0)

    #         c2p1.append(0)
    #         c2p2.append(0)
    #         c2p3.append(0)
    #         c2p123.append(0)

    #         c3p1.append(0)
    #         c3p2.append(1)
    #         c3p3.append(0)
    #         c3p123.append(0)

    #     elif gt == 2 and id == 3:
    #         c1p1.append(0)
    #         c1p2.append(0)
    #         c1p3.append(0)
    #         c1p123.append(0)

    #         c2p1.append(0)
    #         c2p2.append(0)
    #         c2p3.append(0)
    #         c2p123.append(0)

    #         c3p1.append(0)
    #         c3p2.append(0)
    #         c3p3.append(1)
    #         c3p123.append(0)

    #     elif gt == 2 and id == 123:
    #         c1p1.append(0)
    #         c1p2.append(0)
    #         c1p3.append(0)
    #         c1p123.append(0)

    #         c2p1.append(0)
    #         c2p2.append(0)
    #         c2p3.append(0)
    #         c2p123.append(0)

    #         c3p1.append(0)
    #         c3p2.append(0)
    #         c3p3.append(0)
    #         c3p123.append(1)

    # c1p1_count = c1p1.count(1)
    # c1p2_count = c1p2.count(1)
    # c1p3_count = c1p3.count(1)
    # c1p123_count = c1p123.count(1)

    # c2p1_count = c2p1.count(1)
    # c2p2_count = c2p2.count(1)
    # c2p3_count = c2p3.count(1)
    # c2p123_count = c2p123.count(1)

    # c3p1_count = c3p1.count(1)
    # c3p2_count = c3p2.count(1)
    # c3p3_count = c3p3.count(1)
    # c3p123_count = c3p123.count(1)

    # return c1p1_count, c1p2_count, c1p3_count, c1p123_count, c2p1_count, c2p2_count, c2p3_count, c2p123_count, c3p1_count, c3p2_count, c3p3_count, c3p123_count

    gt = np.asarray(gt)
    id_rule = np.asarray(id_rule)

    data = np.column_stack((gt, id_rule))
    id_compare_df = pd.DataFrame(data=data, columns=["ground", "id"])

    c1p1_count = sum( (id_compare_df.ground == 0) & (id_compare_df.id == 1) )
    c1p2_count = sum( (id_compare_df.ground == 0) & (id_compare_df.id == 2) )
    c1p3_count = sum( (id_compare_df.ground == 0) & (id_compare_df.id == 3) )
    c1p123_count = sum( (id_compare_df.ground == 0) & (id_compare_df.id == 123) )

    c2p1_count = sum( (id_compare_df.ground == 1) & (id_compare_df.id == 1) )
    c2p2_count = sum( (id_compare_df.ground == 1) & (id_compare_df.id == 2) )
    c2p3_count = sum( (id_compare_df.ground == 1) & (id_compare_df.id == 3) )
    c2p123_count = sum( (id_compare_df.ground == 1) & (id_compare_df.id == 123) )

    c3p1_count = sum( (id_compare_df.ground == 2) & (id_compare_df.id == 1) )
    c3p2_count = sum( (id_compare_df.ground == 2) & (id_compare_df.id == 2) )
    c3p3_count = sum( (id_compare_df.ground == 2) & (id_compare_df.id == 3) )
    c3p123_count = sum( (id_compare_df.ground == 2) & (id_compare_df.id == 123) )

    return c1p1_count, c1p2_count, c1p3_count, c1p123_count, c2p1_count, c2p2_count, c2p3_count, c2p123_count, c3p1_count, c3p2_count, c3p3_count, c3p123_count

def normalise_data(X):
    print("Calculating the mean values... ", end="", flush=True)
    features_mean = np.mean(X, axis=0)
    print("Done!")

    print("Calculating the std values... ", end="", flush=True)
    features_std = np.std(X, axis=0)
    print("Done!")

    features_std = np.where(features_std != 0, features_std, 1.0)

    print("Normalising the dataset... ", end="", flush=True)
    X = (X - features_mean) / features_std
    print("Done!")

    return X

def create_model():
    # data = []

    X, _, y, feature_names, technique = get_data()

    weights = torch.Tensor(compute_class_weight('balanced', np.unique(y), y))
    weights = weights.to(device)
    
    # dataset split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=100)

    global shape 
    shape = X_train.shape[1]
    
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    X_train = X_train.to(device)
    X_test = X_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)

    # for i in range(len(X_test)):
    #     data.append([X_test[i], y_test[i]])

    # data_loader = torch.utils.data.DataLoader(data, shuffle=False, batch_size=1)

    model = ANN()
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight = weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # net = NeuralNetClassifier(
    #     module=ANN,
    #     criterion=nn.CrossEntropyLoss,
    #     criterion__weight=weights,
    #     optimizer=torch.optim.SGD,
    #     # max_epochs=250,
    #     lr=0.1,
    #     # train_split=skorch.dataset.CVSplit(0.1),
    #     # device='cuda'
    # )

    # # print(X_train.cpu().detach().numpy())
    # # print(y_train.cpu().detach().numpy())

    # # net.fit(X_train.cpu().detach().numpy(), y_train.cpu().detach().numpy())

    # # print(net)

    # # model.eval()
    # y_pred_cv = cross_val_predict(net, X_train.cpu(), y_train.cpu(), cv=5)
    # np.savetxt('./cv_pred.csv', y_pred_cv, delimiter=',')
    # np.savetxt('./y_values.csv', y_train.cpu(), delimiter=',')

    # print(y_pred_cv)
    # exit()

    epochs = 250
    loss_arr = []

    for i in range(epochs):
        y_hat = model(X_train)

        loss = criterion(y_hat, y_train)
        loss_arr.append(loss)
    
        if i % 1 == 0:
            print(f'Epoch: {i} Loss: {loss}')
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # plt.plot(loss_arr)
    # plt.show()

    torch.save(model.state_dict(), './models/model_simple.pt')

    new_choice = input("Would you like to test the model? Yes (1) or no (2). ")

    if new_choice == '1':
        test_model(criterion, optimizer, X_train, y_train, X_test, y_test)
    else:
        exit()

def test_perturbed(X_test, y_test, classifier):
    # model.eval()

    beta = model.output.weight.cpu().detach().numpy()
    bias = model.output.bias.cpu().detach().numpy()

    attack = DeepFool(classifier=classifier)
    df_adv = torch.FloatTensor(attack.generate(x=X_test.cpu())).cuda()

    layer1, layer2, output = model.ec_forward(df_adv)

    y_pred_values, y_pred = torch.max(output.data, dim=1)

    df_acc = accuracy_score(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
    df_cm = confusion_matrix(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
    print("Testing accuracy: " + str(df_acc * 100) + "%")
    print("DeepFool: ", df_cm)
    np.savetxt('./cm_files/deepfool_confusion_matrix.csv', np.transpose(np.asarray(df_cm)), delimiter=',')

    mass_list = []

    warnings.filterwarnings("ignore")

    mass_list, mass_function_1, mass_function_2, mass_function_3, mass_function_12, mass_function_13, mass_function_23, mass_function_123, conflict, eta, eta_plus, eta_neg = calculate_masses(layer2, beta, bias, y_test)

    np.savetxt('./masses/deepfool_mass_function_1.csv', mass_function_1, delimiter=',')
    np.savetxt('./masses/deepfool_mass_function_2.csv', mass_function_2, delimiter=',')
    np.savetxt('./masses/deepfool_mass_function_3.csv', mass_function_3, delimiter=',')
    np.savetxt('./masses/deepfool_mass_function_12.csv', mass_function_12, delimiter=',')
    np.savetxt('./masses/deepfool_mass_function_13.csv', mass_function_13, delimiter=',')
    np.savetxt('./masses/deepfool_mass_function_23.csv', mass_function_23, delimiter=',')
    np.savetxt('./masses/deepfool_mass_function_123.csv', mass_function_123, delimiter=',')
    np.savetxt('./masses/deepfool_conflict.csv', conflict, delimiter=',')
    np.savetxt('./masses/deepfool_X.csv', df_adv.cpu().detach().numpy(), delimiter=',')
    np.savetxt('./masses/deepfool_eta.csv', eta, delimiter=',')
    np.savetxt('./masses/deepfool_eta_plus.csv', eta_plus, delimiter=',')
    np.savetxt('./masses/deepfool_eta_neg.csv', eta_neg, delimiter=',')
    np.savetxt('./masses/deepfool_layer2.csv', layer2.cpu().detach().numpy(), delimiter=',')
    np.savetxt('./masses/deepfool_bias.csv', bias, delimiter=',')
    np.savetxt('./masses/deepfool_beta.csv', beta, delimiter=',')

    print("Joining mass functions... ", end="", flush=True)
    mass_functions = join_mass_functions(mass_function_1, mass_function_2, mass_function_3, mass_function_12, mass_function_13, mass_function_23, mass_function_123)
    print("Done!")

    print("Calculating beliefs... ", end="", flush=True)
    t1_belief_array, t2_belief_array, t3_belief_array = calculate_belief(mass_functions)
    print("Done!")

    print("Calculating plausiblities... ", end="", flush=True)
    t1_plausibility_array, t2_plausibility_array, t3_plausibility_array = calculate_plausibility(mass_functions)
    print("Done!")

    print("Calculating Maximum Plausibility Rule... ", end="", flush=True)
    decision_mp = calculate_mp_rule(t1_plausibility_array, t2_plausibility_array, t3_plausibility_array)
    print("Done!")

    print("Calculating Conservative Approach Rule... ", end="", flush=True)
    decision_ca = calculate_ca_rule(t1_plausibility_array, t2_plausibility_array, t3_plausibility_array, t1_belief_array, t2_belief_array, t3_belief_array)
    print("Done!")

    print("Calculating Maximal Element Rule... ", end="", flush=True)
    decision_me = calculate_me_rule(t1_plausibility_array, t2_plausibility_array, t3_plausibility_array, t1_belief_array, t2_belief_array, t3_belief_array)
    print("Done!")

    print("Calculating Interval Dominance Rule... ", end="", flush=True)
    decision_id = calculate_id_rule(decision_me, decision_ca)
    print("Done!")

    np.savetxt('./individual_files/deepfool_ca.csv', np.asarray(decision_ca), delimiter=',')
    np.savetxt('./individual_files/deepfool_me.csv', np.asarray(decision_me), delimiter=',')
    np.savetxt('./individual_files/deepfool_id.csv', np.asarray(decision_id), delimiter=',')
    np.savetxt('./individual_files/deepfool_mp.csv', np.asarray(decision_mp), delimiter=',')
    np.savetxt('./individual_files/deepfool_y_test.csv', y_test.cpu().detach().numpy(), delimiter=',')
    np.savetxt('./individual_files/deepfool_y_pred.csv', y_pred.cpu().detach().numpy(), delimiter=',')

    c1p1_count, c1p2_count, c1p3_count, c1p123_count, c2p1_count, c2p2_count, c2p3_count, c2p123_count, c3p1_count, c3p2_count, c3p3_count, c3p123_count = tabulate_id_rule(decision_id, y_test.cpu().detach().numpy())

    total_num = len(y_test.cpu().detach().numpy())

    c1p1_perc = (c1p1_count / total_num) * 100
    c1p2_perc = (c1p2_count / total_num) * 100
    c1p3_perc = (c1p3_count / total_num) * 100
    c1p123_perc = (c1p123_count / total_num) * 100

    c2p1_perc = (c2p1_count / total_num) * 100
    c2p2_perc = (c2p2_count / total_num) * 100
    c2p3_perc = (c2p3_count / total_num) * 100
    c2p123_perc = (c2p123_count / total_num) * 100

    c3p1_perc = (c3p1_count / total_num) * 100
    c3p2_perc = (c3p2_count / total_num) * 100
    c3p3_perc = (c3p3_count / total_num) * 100
    c3p123_perc = (c3p123_count / total_num) * 100


    cm_file = open("cm_files/deepfool_id_rule_cm.txt", "w")

    cm1 = [
        ["", "1", "2", "3"],
        ["1", np.round(c1p1_perc, 2), np.round(c2p1_perc, 2), np.round(c3p1_perc, 2)],
        ["2", np.round(c1p2_perc, 2), np.round(c2p2_perc, 2), np.round(c3p2_perc, 2)],
        ["3", np.round(c1p3_perc, 2), np.round(c2p3_perc, 2), np.round(c3p3_perc, 2)],
        ["Uncertain", np.round(c1p123_perc, 2), np.round(c2p123_perc, 2), np.round(c3p123_perc, 2)]
    ]

    cm2 = [
            ["", "1", "2", "3"],
            ["1", c1p1_count, c2p1_count, c3p1_count],
            ["2", c1p2_count, c2p2_count, c3p2_count],
            ["3", c1p3_count, c2p3_count, c3p3_count],
            ["Uncertain", c1p123_count, c2p123_count, c3p123_count]
        ]

    cm_file.write(tabulate(cm1))
    cm_file.write("\n\n")
    cm_file.write(tabulate(cm2))
    cm_file.close() 



    jsma_attack = SaliencyMapMethod(classifier=classifier)
    jsma_adv = jsma_attack.generate(x=X_test.cpu())

    layer1, layer2, output = model.ec_forward(torch.from_numpy(jsma_adv).cuda())
    y_pred_values, y_pred = torch.max(output.data, dim=1)

    acc = accuracy_score(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
    jsma_cm = confusion_matrix(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
    print("Testing accuracy: " + str(acc * 100) + "%")
    print("JSMA: ", jsma_cm)
    np.savetxt('./cm_files/jsma_confusion_matrix.csv', np.transpose(np.asarray(jsma_cm)), delimiter=',')

    mass_list = []

    warnings.filterwarnings("ignore")

    mass_list, mass_function_1, mass_function_2, mass_function_3, mass_function_12, mass_function_13, mass_function_23, mass_function_123, conflict,eta, eta_plus, eta_neg = calculate_masses(layer2, beta, bias, y_test)

    np.savetxt('./masses/jsma_mass_function_1.csv', mass_function_1, delimiter=',')
    np.savetxt('./masses/jsma_mass_function_2.csv', mass_function_2, delimiter=',')
    np.savetxt('./masses/jsma_mass_function_3.csv', mass_function_3, delimiter=',')
    np.savetxt('./masses/jsma_mass_function_12.csv', mass_function_12, delimiter=',')
    np.savetxt('./masses/jsma_mass_function_13.csv', mass_function_13, delimiter=',')
    np.savetxt('./masses/jsma_mass_function_23.csv', mass_function_23, delimiter=',')
    np.savetxt('./masses/jsma_mass_function_123.csv', mass_function_123, delimiter=',')
    np.savetxt('./masses/jsma_conflict.csv', conflict, delimiter=',')
    np.savetxt('./masses/jsma_X.csv', jsma_adv.cpu().detach().numpy(), delimiter=',')

    print("Joining mass functions... ", end="", flush=True)
    mass_functions = join_mass_functions(mass_function_1, mass_function_2, mass_function_3, mass_function_12, mass_function_13, mass_function_23, mass_function_123)
    print("Done!")

    print("Calculating beliefs... ", end="", flush=True)
    t1_belief_array, t2_belief_array, t3_belief_array = calculate_belief(mass_functions)
    print("Done!")

    print("Calculating plausiblities... ", end="", flush=True)
    t1_plausibility_array, t2_plausibility_array, t3_plausibility_array = calculate_plausibility(mass_functions)
    print("Done!")

    print("Calculating Maximum Plausibility Rule... ", end="", flush=True)
    decision_mp = calculate_mp_rule(t1_plausibility_array, t2_plausibility_array, t3_plausibility_array)
    print("Done!")

    print("Calculating Conservative Approach Rule... ", end="", flush=True)
    decision_ca = calculate_ca_rule(t1_plausibility_array, t2_plausibility_array, t3_plausibility_array, t1_belief_array, t2_belief_array, t3_belief_array)
    print("Done!")

    print("Calculating Maximal Element Rule... ", end="", flush=True)
    decision_me = calculate_me_rule(t1_plausibility_array, t2_plausibility_array, t3_plausibility_array, t1_belief_array, t2_belief_array, t3_belief_array)
    print("Done!")

    print("Calculating Interval Dominance Rule... ", end="", flush=True)
    decision_id = calculate_id_rule(decision_me, decision_ca)
    print("Done!")

    np.savetxt('./individual_files/jsma_ca.csv', np.asarray(decision_ca), delimiter=',')
    np.savetxt('./individual_files/jsma_me.csv', np.asarray(decision_me), delimiter=',')
    np.savetxt('./individual_files/jsma_id.csv', np.asarray(decision_id), delimiter=',')
    np.savetxt('./individual_files/jsma_mp.csv', np.asarray(decision_mp), delimiter=',')
    np.savetxt('./individual_files/jsma_y_test.csv', y_test.cpu().detach().numpy(), delimiter=',')
    np.savetxt('./individual_files/jsma_y_pred.csv', y_pred.cpu().detach().numpy(), delimiter=',')

    c1p1_count, c1p2_count, c1p3_count, c1p123_count, c2p1_count, c2p2_count, c2p3_count, c2p123_count, c3p1_count, c3p2_count, c3p3_count, c3p123_count = tabulate_id_rule(decision_id, y_test.cpu().detach().numpy())

    total_num = len(y_test.cpu().detach().numpy())

    c1p1_perc = (c1p1_count / total_num) * 100
    c1p2_perc = (c1p2_count / total_num) * 100
    c1p3_perc = (c1p3_count / total_num) * 100
    c1p123_perc = (c1p123_count / total_num) * 100

    c2p1_perc = (c2p1_count / total_num) * 100
    c2p2_perc = (c2p2_count / total_num) * 100
    c2p3_perc = (c2p3_count / total_num) * 100
    c2p123_perc = (c2p123_count / total_num) * 100

    c3p1_perc = (c3p1_count / total_num) * 100
    c3p2_perc = (c3p2_count / total_num) * 100
    c3p3_perc = (c3p3_count / total_num) * 100
    c3p123_perc = (c3p123_count / total_num) * 100


    cm_file = open("cm_files/jsma_id_rule_cm.txt", "w")

    cm1 = [
        ["", "1", "2", "3"],
        ["1", np.round(c1p1_perc, 2), np.round(c2p1_perc, 2), np.round(c3p1_perc, 2)],
        ["2", np.round(c1p2_perc, 2), np.round(c2p2_perc, 2), np.round(c3p2_perc, 2)],
        ["3", np.round(c1p3_perc, 2), np.round(c2p3_perc, 2), np.round(c3p3_perc, 2)],
        ["Uncertain", np.round(c1p123_perc, 2), np.round(c2p123_perc, 2), np.round(c3p123_perc, 2)]
    ]

    cm2 = [
            ["", "1", "2", "3"],
            ["1", c1p1_count, c2p1_count, c3p1_count],
            ["2", c1p2_count, c2p2_count, c3p2_count],
            ["3", c1p3_count, c2p3_count, c3p3_count],
            ["Uncertain", c1p123_count, c2p123_count, c3p123_count]
        ]

    cm_file.write(tabulate(cm1))
    cm_file.write("\n\n")
    cm_file.write(tabulate(cm2))
    cm_file.close() 



def test_model(criterion, optimizer, X_train, y_train, X_test, y_test):
    global model
    model = ANN()

    # model.eval()

    model.load_state_dict(torch.load('./models/model_simple.pt'))

    classifier = PyTorchClassifier(
        model = model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(shape),
        nb_classes=3,
    )

    # classifier.fit(X_train.cpu(), y_train.cpu(), batch_size=1)

    beta = model.output.weight.cpu().detach().numpy()
    bias = model.output.bias.cpu().detach().numpy()

    layer1, layer2, output = model.ec_forward(X_test)

    y_pred_values, y_pred = torch.max(output.data, dim=1)

    acc = accuracy_score(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
    unperturbed_cm = confusion_matrix(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
    print("Testing accuracy: " + str(acc * 100) + "%")
    print("Unperturbed", unperturbed_cm)
    np.savetxt('./cm_files/unperturbed_confusion_matrix.csv', np.transpose(np.asarray(unperturbed_cm)), delimiter=',')

    mass_list = []

    warnings.filterwarnings("ignore")

    mass_list, mass_function_1, mass_function_2, mass_function_3, mass_function_12, mass_function_13, mass_function_23, mass_function_123, conflict, eta, eta_plus, eta_neg = calculate_masses(layer2, beta, bias, y_test)

    np.savetxt('./masses/unperturbed_mass_function_1.csv', mass_function_1, delimiter=',')
    np.savetxt('./masses/unperturbed_mass_function_2.csv', mass_function_2, delimiter=',')
    np.savetxt('./masses/unperturbed_mass_function_3.csv', mass_function_3, delimiter=',')
    np.savetxt('./masses/unperturbed_mass_function_12.csv', mass_function_12, delimiter=',')
    np.savetxt('./masses/unperturbed_mass_function_13.csv', mass_function_13, delimiter=',')
    np.savetxt('./masses/unperturbed_mass_function_23.csv', mass_function_23, delimiter=',')
    np.savetxt('./masses/unperturbed_mass_function_123.csv', mass_function_123, delimiter=',')
    np.savetxt('./masses/unperturbed_conflict.csv', conflict, delimiter=',')

    # np.savetxt('./masses/unperturbed_original_data_file.csv', mass_list, delimiter=',')

    print("Joining mass functions... ", end="", flush=True)
    mass_functions = join_mass_functions(mass_function_1, mass_function_2, mass_function_3, mass_function_12, mass_function_13, mass_function_23, mass_function_123)
    print("Done!")

    print("Calculating beliefs... ", end="", flush=True)
    t1_belief_array, t2_belief_array, t3_belief_array = calculate_belief(mass_functions)
    print("Done!")

    print("Calculating plausiblities... ", end="", flush=True)
    t1_plausibility_array, t2_plausibility_array, t3_plausibility_array = calculate_plausibility(mass_functions)
    print("Done!")

    print("Calculating Maximum Plausibility Rule... ", end="", flush=True)
    decision_mp = calculate_mp_rule(t1_plausibility_array, t2_plausibility_array, t3_plausibility_array)
    print("Done!")

    print("Calculating Conservative Approach Rule... ", end="", flush=True)
    decision_ca = calculate_ca_rule(t1_plausibility_array, t2_plausibility_array, t3_plausibility_array, t1_belief_array, t2_belief_array, t3_belief_array)
    print("Done!")

    print("Calculating Maximal Element Rule... ", end="", flush=True)
    decision_me = calculate_me_rule(t1_plausibility_array, t2_plausibility_array, t3_plausibility_array, t1_belief_array, t2_belief_array, t3_belief_array)
    print("Done!")

    print("Calculating Interval Dominance Rule... ", end="", flush=True)
    decision_id = calculate_id_rule(decision_me, decision_ca)
    print("Done!")

    np.savetxt('./individual_files/unperturbed_ca.csv', np.asarray(decision_ca), delimiter=',')
    np.savetxt('./individual_files/unperturbed_me.csv', np.asarray(decision_me), delimiter=',')
    np.savetxt('./individual_files/unperturbed_id.csv', np.asarray(decision_id), delimiter=',')
    np.savetxt('./individual_files/unperturbed_mp.csv', np.asarray(decision_mp), delimiter=',')
    np.savetxt('./individual_files/unperturbed_X.csv', X_test.cpu().detach().numpy(), delimiter=',')
    np.savetxt('./individual_files/unperturbed_y.csv', y_test.cpu().detach().numpy(), delimiter=',')
    np.savetxt('./individual_files/unperturbed_y_pred.csv', y_pred.cpu().detach().numpy(), delimiter=',')


    c1p1_count, c1p2_count, c1p3_count, c1p123_count, c2p1_count, c2p2_count, c2p3_count, c2p123_count, c3p1_count, c3p2_count, c3p3_count, c3p123_count = tabulate_id_rule(decision_id, y_test.cpu().detach().numpy())

    total_num = len(y_test.cpu().detach().numpy())

    c1p1_perc = (c1p1_count / total_num) * 100
    c1p2_perc = (c1p2_count / total_num) * 100
    c1p3_perc = (c1p3_count / total_num) * 100
    c1p123_perc = (c1p123_count / total_num) * 100

    c2p1_perc = (c2p1_count / total_num) * 100
    c2p2_perc = (c2p2_count / total_num) * 100
    c2p3_perc = (c2p3_count / total_num) * 100
    c2p123_perc = (c2p123_count / total_num) * 100

    c3p1_perc = (c3p1_count / total_num) * 100
    c3p2_perc = (c3p2_count / total_num) * 100
    c3p3_perc = (c3p3_count / total_num) * 100
    c3p123_perc = (c3p123_count / total_num) * 100


    cm_file = open("cm_files/unperturbed_id_rule_cm.txt", "w")

    cm1 = [
        ["", "1", "2", "3"],
        ["1", np.round(c1p1_perc, 2), np.round(c2p1_perc, 2), np.round(c3p1_perc, 2)],
        ["2", np.round(c1p2_perc, 2), np.round(c2p2_perc, 2), np.round(c3p2_perc, 2)],
        ["3", np.round(c1p3_perc, 2), np.round(c2p3_perc, 2), np.round(c3p3_perc, 2)],
        ["Uncertain", np.round(c1p123_perc, 2), np.round(c2p123_perc, 2), np.round(c3p123_perc, 2)]
    ]

    cm2 = [
            ["", "1", "2", "3"],
            ["1", c1p1_count, c2p1_count, c3p1_count],
            ["2", c1p2_count, c2p2_count, c3p2_count],
            ["3", c1p3_count, c2p3_count, c3p3_count],
            ["Uncertain", c1p123_count, c2p123_count, c3p123_count]
        ]

    cm_file.write(tabulate(cm1))
    cm_file.write("\n\n")
    cm_file.write(tabulate(cm2))
    cm_file.close() 


    next_choice = input("Would you like to test on perturbed data? Yes (1), or no (2). ")
    
    if next_choice == '1':
        test_perturbed(X_test, y_test, classifier)
    else:
        exit()

# X, y = make_blobs(n_samples=1000, random_state=0, centers=[(-0.5, -0.5), (-0.5, -0.5), (10, -10)], cluster_std=[5, 2, 5], n_features=2)
# X = normalise_data(X)

# df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
# colors = {0:'red', 1:'blue', 2:'green'}
# fig, ax = plt.subplots()
# grouped = df.groupby('label')
# for key, group in grouped:
#     group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
# plt.show()

class ANN(nn.Module):
    def __init__(self):
        super().__init__()

        self.h1 = nn.Linear(in_features=shape, out_features=20)
        self.dropout = nn.Dropout(p=0.5)
        self.h2 = nn.Linear(in_features=20, out_features=10)
        self.output = nn.Linear(in_features=10, out_features=3)

    def forward(self, x):
        layer1 = F.relu(self.h1(x))
        layer1 = self.dropout(layer1)
        layer2 = F.relu(self.h2(layer1))
        output = torch.softmax(self.output(layer2), dim=1)
        return output
    
    def ec_forward(self, x):
        layer1 = F.relu(self.h1(x))
        layer1 = self.dropout(layer1)
        layer2 = F.relu(self.h2(layer1))
        output = torch.softmax(self.output(layer2), dim=1)
        return layer1, layer2, output

torch.manual_seed(0)
np.random.seed(0)
# torch.use_deterministic_algorithms(True)

shape = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

folders = ['models', 'cm_files', 'masses', 'individual_files']

for folder in folders:
    if os.path.exists('./' + folder + '/'):
        shutil.rmtree('./' + folder + '/')

    if not os.path.exists('./' + folder + '/'):
        os.mkdir('./' + folder + '/')

input_choice = input("""
                      1: Create NN model
                      2: Test NN model
                      3: Quit

                      Please enter your choice: """)

if input_choice == '1':
    create_model()

elif input_choice == '2':
    test_model()




exit()


# v1 = 0.0
# v2 = 0.0


# while v1 is not "q" or v2 is not "q":
#     model.eval()

#     v1 = input("Enter v1: ")
#     v2 = input("Enter v2: ")
#     y_test = 0
    
#     values = torch.FloatTensor([float(v1), float(v2)])
#     values = values.unsqueeze(0)

#     with torch.no_grad():
#         layer1, layer2, output = model.ec_forward(values)
    
#     # phi = layer2

#     mass_list, mass_function_1, mass_function_2, mass_function_3, mass_function_12, mass_function_13, mass_function_23, mass_function_123 = calculate_masses(layer2, beta, bias, y_test)
#     mass_functions = join_mass_functions(mass_function_1, mass_function_2, mass_function_3, mass_function_12, mass_function_13, mass_function_23, mass_function_123)
#     t1_belief_array, t2_belief_array, t3_belief_array = calculate_belief(mass_functions)
#     t1_plausibility_array, t2_plausibility_array, t3_plausibility_array = calculate_plausibility(mass_functions)
#     decision_mp = calculate_mp_rule(t1_plausibility_array, t2_plausibility_array, t3_plausibility_array)
#     decision_ca = calculate_ca_rule(t1_plausibility_array, t2_plausibility_array, t3_plausibility_array, t1_belief_array, t2_belief_array, t3_belief_array)
#     decision_me = calculate_me_rule(t1_plausibility_array, t2_plausibility_array, t3_plausibility_array, t1_belief_array, t2_belief_array, t3_belief_array)
#     decision_id = calculate_id_rule(decision_me, decision_ca)

#     # print("Layer 2", layer2)
#     print("MP Rule", decision_mp)
#     print("ID Rule", decision_id)
#     print()

# sys.exit()


# print(c1p1_count, c2p1_count, c3p1_count)
# print(c1p2_count, c2p2_count, c3p2_count)
# print(c1p3_count, c2p3_count, c3p3_Count)
# print(c1p123_count, c2p123_count, c3p123_count)

# sys.exit()