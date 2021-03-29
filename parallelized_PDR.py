#!/usr/bin/bash/env python
from parallel_dim_reduction import *
import numpy as np
import matplotlib.pyplot as plt
from plots_and_figs import *
from tables import *
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=NaturalNameWarning)
warnings.filterwarnings("ignore", category=NaturalNameWarning)

def PDR_algo(path_to_data, writer, extraspec, nK, nL, alph, num_runs, samp_freq, stop_search, max_iter):
    
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
                      "sg_met": loader.sg_met, "runID": runID, "extra_spec": extraspec}
        
        if param_dict["extra_spec"] is None:
            param_dict.pop("extra_spec")

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
        
def PDR_alph_range(path_to_data, writer, extraspec, nK, nL, alph_arr, num_runs, samp_freq, stop_search, max_iter):
    for alph in alph_arr:
        alph = float('%.3f' % alph)
        PDR_algo(path_to_data, writer, extraspec, nK, nL, alph, num_runs, samp_freq, stop_search, max_iter)
        
def init_file_struct(path_to_dataset, naming_dict):
    parent_dir = lambda: os.chdir(os.path.dirname(os.getcwd()))
    
    dataset_dir, dataset_name = pop_path(path_to_dataset)
    parent_dir()
    cd = os.getcwd()
    os.chdir(dataset_dir)
    parent_dir()
    path_to_data_dir = os.getcwd()
    os.chdir(cd)
    dataset = os.path.splitext(dataset_name)[0]
    
    naming_dict_cpy = naming_dict.copy()
    if naming_dict["extra_spec"] is None:
        naming_dict_cpy.pop("extra_spec")

    hdf_filename = ''
    for key in list(naming_dict_cpy.keys()):
        hdf_filename += naming_dict_cpy[key] + "_"
    hdf_filename += ".h5"
    path_to_hdf = os.path.join(path_to_data_dir,
                               dataset,
                               *[naming_dict_cpy[key] for key in naming_dict_cpy.keys()],
                               hdf_filename)

    p, f = pop_path(path_to_hdf)
    os.makedirs(p, exist_ok=True)
    
    return path_to_hdf

def step_arr(start, stop, step):
    arr = np.linspace(start, stop, int((stop - start)/step + 1))
    for i in range(len(arr)):
        arr[i] = float('%.3f' % arr[i])
    return arr

#---------------------------------------------------------------------

alph_arr = step_arr(0.05, 0.30, 0.01)
nL = 2
nK = 11
num_runs = 2

naming_dict = {"model_type": 'testrun',
               "extra_spec": None,
               "nK": "nK=" + str(nK),
               "nL": "nL=" + str(nL)}

path_to_dataset = "/blue/pdixit/lukasherron/parallel_dim_reduction/data/datasets/bovine_data.mat"
path_to_hdf = init_file_struct(path_to_dataset, naming_dict)

writer = DataWriter(path_to_hdf, mode="w")
PDR_alph_range(path_to_dataset, writer, naming_dict["extra_spec"], nK, nL, alph_arr, num_runs, 500, 1000, 2000)
writer.close()