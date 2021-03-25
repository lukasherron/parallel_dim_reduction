#!/usr/bin/bash/env python
from parallel_dim_reduction import *
import numpy as np
import matplotlib.pyplot as plt
from plots_and_figs import *
from tables import *
import sys


def PDR_algo(path_to_data_dir, writer, nK, nL, alph, num_runs, samp_freq, stop_search, max_iter):
    
    for runID in range(num_runs):
        
        # LOADING DATA
        loader = DataLoader()
        xs, metx, labels = loader.load_data(path_to_data)
        train_size = 0.75
        test_size = 1 - train_size
        val_size = 0
        idx_train, idx_test, idx_val = loader.split_data(test_size=test_size, val_size=val_size)
        metz_train, metz_test, metz_val = loader.preprocess_metadata(metx, idx_train, idx_test, idx_val)

        # INITIALIZING DICTS AND OBJECTS
        param_dict = {"nK": nK, "alph": alph, "etaZ": 3e-5, "etaT": 3e-5, "etaC": 1e-5, "nPC": nK - nL, "nL": nL,
                      "nMet": loader.nMet, "nD": xs.shape[1], "nSamp": loader.nSamp, "mu_met": loader.mu_met,
                      "sg_met": loader.sg_met, "runID": runID}

        eval_dict = dict(e1=np.nan, e2=np.nan, KL_final=np.nan, KL_train=[0], KL_val=[0])

        grad_dict = dict(grz=np.nan, grthet=np.nan, grc=np.nan)

        data_dict = {"xs_train": xs[idx_train], "xs_test": xs[idx_test], "xs_val": xs[idx_val], "Q": np.nan,
                     "metz_train": metz_train, "metz_test": metz_test, "metz_val": metz_val, "C": np.nan, "Z": None,
                     "Zcon": np.nan, "Zfr": np.nan,
                     "thet": np.random.random(size=(param_dict["nPC"] + param_dict["nL"], param_dict["nD"])),
                     "labels": labels, "idx_train": idx_train, "idx_test": idx_test, "idx_val": idx_val}
        
        
        ParamObj = ObjFromDict(param_dict, "param")
        EvalObj = ObjFromDict(eval_dict, "eval")
        GradObj = ObjFromDict(grad_dict, "grad")
        DataObj = ObjFromDict(data_dict, "data")

    # TRAINING
        # PCA is a good initial guess for PDR
        PCA_red = PCA_reduction(ParamObj, DataObj, EvalObj)
        PCA_red.dim_reduction(guess=True)
        DataObj, EvalObj = PCA_red.finalize()

        # Parallel Dimensionality Reduction
        PDR = Parallel_Dimensionality_Reduction(ParamObj, DataObj, EvalObj, GradObj)
        PDR.gradient_descent(writer, samp_freq, stop_search, max_iter, validate=False)
        PDR.eval_training_performance()
        DataObj, EvalObj, GradObj = PDR.finalize()
        
    # TESTING PHASE
        # Making predictions
        PFM = Predict_From_Model()
        PFM.inherit_model(ParamObj, DataObj, EvalObj)
        PFM.predict_microbiome()
        PredObj = PFM.finalize()
 
        writer.write_obj(ParamObj, ParamObj)
        writer.write_obj(ParamObj, EvalObj)
        writer.write_obj(ParamObj, DataObj)
        writer.write_obj(ParamObj, PredObj)
        
    writer.close()

def PDR_alph_range(path_to_data, writer, nK, nL, alph_arr, num_runs, samp_freq, stop_search, max_iter):
    for alph in alph_arr:
        alph = float('%.3f' % alph)
        PDR_algo(path_to_data, writer, nK, nL, alph, num_runs, samp_freq, stop_search, max_iter)

#-------------------------------------------------------------------------------------------------------     

alph_arr = [0.05]
path_to_data = "/blue/pdixit/lukasherron/parallel_dim_reduction/data/datasets/bovine_data.mat"
model_type='test_run'
further_spec=None

nL = 2
nK = 12
num_runs = 2

path_to_data_dir = "/blue/pdixit/lukasherron/parallel_dim_reduction/data/"
dataset_name = create_filetree(path_to_data_dir, path_to_data, model_type, further_spec=further_spec)
fs = str(further_spec)
if fs == "None":
    fs = ""
hdf_filename = model_type + "_nK=" + str(nK) + "_nL=" + str(nL) + "_h5file_.h5"
path_to_hdf = os.path.join("/blue/pdixit/lukasherron/parallel_dim_reduction/data/bovine_data/", 
                           model_type,
                           "nK="+str(nK),
                           "nL="+str(nL),
                          hdf_filename)

writer = DataWriter(path_to_hdf, mode="w")
PDR_alph_range(path_to_data, writer, nK, nL, alph_arr, num_runs, samp_freq=500, stop_search=1000, max_iter=2000)