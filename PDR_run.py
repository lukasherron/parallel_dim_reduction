#!/usr/bin/env python
from parallel_dim_reduction import *
import numpy as np
import matplotlib.pyplot as plt
from plots_and_figs import *

# LOADING DATA
loader = DataLoader()
xs, metx, labels = loader.load_data("/home/lukasherron/PycharmProjects/pythonProject/combined_data.mat")
idx_train, idx_test = loader.split_data(test_size=0.2)
metz_train, metz_test = loader.preprocess_metadata(metx, idx_train, idx_test)

# INITIALIZING DICTS AND OBJECTS

param_dict = {
    "kks": [7, 12, 17, 22, 27],
    "e10": [2249, 1129, 660, 457, 311],
    "e20": [168, 129, 85, 70],
    "idx": 2,
    "alph": None,
    "etaZ": 0.0001,
    "etaT": 0.0001,
    "etaC": 0.0001,
    "nPC": None,
    "nL": 2,
    "nMet": loader.nMet,
    "nD": xs.shape[1],
    "nSamp": loader.nSamp,
    "mu_met": loader.mu_met,
    "sg_met": loader.sg_met
}

# Calculating alph and nPC
idx = param_dict["idx"]
e20 = param_dict["e20"]
e10 = param_dict["e10"]
kks = param_dict["kks"]
nL = param_dict["nL"]

param_dict["alph"] = e20[idx] / (e20[idx] + e10[idx])
param_dict["nPC"] = kks[idx] - nL

eval_dict = {
    "e1":       None,
    "e2":       None,
    "KL_final": None,
    "KL_train": [],
    "KL_val":   []

}

grad_dict = {
    "grz":      None,
    "grthet":   None,
    "grc":      None

}

data_dict = {
    "xs_train":     xs[idx_train],
    "xs_test":      xs[idx_test],
    "Q":            None,
    "metz_train":   metz_train,
    "metz_test":    metz_test,
    "C":            None,
    "Z":            None,
    "Zcon":         None,
    "Zfr":          None,
    "thet":         np.random.random(size=(param_dict["nPC"] + param_dict["nL"], param_dict["nD"])),
    "labels":       labels,
    "idx_train":    idx_train,
    "idx_test":     idx_test
}
#%%
print("started...")
path = "/home/lukasherron/PycharmProjects/pythonProject/model_nK=" + str(kks[idx]) + "_nL=" + str(nL) + ".xlsx"
writer = DataWriter()
writer.create_model_workbook(path, data_dict)

# TRAINING

# PCA is a good initial guess for PDR
PCA_red = PCA_reduction(param_dict, data_dict, eval_dict)
PCA_red.dim_reduction(guess=True)
data_dict, eval_dict = PCA_red.finalize()

# Parallel Dimensionality Reduction
PDR = Parallel_Dimensionality_Reduction(param_dict, data_dict, eval_dict, grad_dict)
PDR.gradient_descent(path, validate=True)
PDR.eval_training_performance()
data_dict, eval_dict, grad_dict = PDR.finalize()  # Model is saved
#%%
# TESTING PHASE
randn = str(np.random.randint(0,100)).zfill(3)
path_to_model = "/home/lukasherron/PycharmProjects/pythonProject/model_nK=" + str(kks[idx]) + "_nL=" + str(nL) + ".xlsx"
path_to_pred = "/home/lukasherron/PycharmProjects/pythonProject/model_nK=" + str(kks[idx]) + "_nL=" + str(nL) + "_pred.xlsx"
path_to_data = "/home/lukasherron/PycharmProjects/pythonProject/combined_data.mat"
# Loading model
print("loading model ...")
model_param_dict, model_eval_dict, model_data_dict = loader.load_model(path_to_model)
# Making predictions
print("making predictions ...")
PFM = Predict_From_Model()
PFM.load_model(path_to_model)
PFM.predict_microbiome(path_to_model=path_to_model)
pred_dict = PFM.finalize()
# Saving predictions
print("saving predictions")
writer.create_pred_workbook(path_to_pred, pred_dict)
writer.write_pred_workbook(path_to_pred, pred_dict)
#%%
model_names = ["model_1", "model_2", "model_3"]
path_arr = [
    "/home/lukasherron/PycharmProjects/pythonProject/model_nK=12_nL=2.xlsx",
    "/home/lukasherron/PycharmProjects/pythonProject/model_nK=17_nL=2.xlsx",
    "/home/lukasherron/PycharmProjects/pythonProject/model_nK=22_nL=2.xlsx"
]
path_dict = {model_names[i]: path_arr[i] for i in range(len(model_names))}
model_dict = {model_names[i]: None for i in range(len(model_names))}

for idx, key in list(enumerate(path_dict.keys())):
    loader = DataLoader()
    model_param_dict, model_eval_dict, model_data_dict = loader.load_model(path_dict[key])
    model_dict[key] = [model_param_dict, model_eval_dict, model_data_dict]



KL_train_dict = {name: None for name in model_names}
KL_val_dict = {name: None for name in model_names}

for idx, key in list(enumerate(model_dict.keys())):
    KL_train_dict[key] = model_dict[key][1]["KL_train"]
    KL_val_dict[key] = model_dict[key][1]["KL_val"]

#%%
KL_tot = np.array(KL_train_dict['model_2']) + np.array(KL_val_dict['model_2'])
KL_tot = gaussian_filter1d(KL_tot, 5)

KL_p = KL_tot[1:-1] - KL_tot[0:-2]
KL_pp = KL_p[1:-1] - KL_p[0:-2]

fig_1 = plt.figure()
plt.plot(np.arange(332), KL_pp)
plt.title("KL_pp")
plt.savefig("K_pp.jpg")

fig_2 = plt.figure()
plt.plot(np.arange(334), KL_p)
plt.title("K_p")
plt.savefig("K_p.jpg")

#%%
Q_test = []
for x in pred_dict["Q"].ravel():
    if x > 1e-5:
        Q_test.append(x)
    else:
        Q_test.append(0)

my_scatter(-np.ma.log10(Q_test), -np.ma.log10(pred_dict["xs_test"].ravel()), "Q (test) (log)", "xs (test) (log)",

           "Prediction Accuracy (nK = 17, nL = 2)", fig_size=(7, 7), x_lim=5, y_lim=5, save=True, save_path=
           "predictions_accuracy_log_nK=17_nL=2.jpg")

#%%
Q_train = []
for x in model_data_dict["Q"].ravel():
    if x > 1e-5:
        Q_train.append(x)
    else:
        Q_train.append(0)
my_scatter(-np.ma.log10(Q_train), -np.ma.log10(model_data_dict["xs_train"].ravel()), "Q (train) (log)", "xs (train) (log)",
           "Training Accuracy (nK = 17, nL = 2)", fig_size=(7, 7), x_lim=5, y_lim=5, save=True, save_path=
           "training_accuracy_log_nK=17_nL=2.jpg")


