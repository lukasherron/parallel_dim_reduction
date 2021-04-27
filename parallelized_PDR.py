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
warnings.filterwarnings("ignore", category=FlavorWarning)

def PDR_algo(paths_to_data, writer, extraspec, nK, nL, alph, num_runs, samp_freq, stop_search, max_iter,
             farm_train, farm_test):
    
    [path_to_metadata, path_to_xs] = paths_to_data
    for runID in range(num_runs):
        
        # LOADING DATA
        loader = DataLoader()
        if farm_train == farm_test:
            
            metx, labels, idx = loader.load_csv(path_to_metadata, farm=farm_train)
            xs, _, _ = loader.load_csv(path_to_xs, idx=idx)
            train_size = 0.80
            test_size = 0.20
            val_size = 0
            [loader.nSamp , nMet] = metx.shape
            idx_train, idx_test, idx_val = loader.split_data(test_size=test_size, val_size=val_size)
            metz_train, metz_test, metz_val = loader.preprocess_metadata(metx, idx_train, idx_test, idx_val)
            xs_test = xs[idx_test]
            xs_val = xs[idx_val]
            xs_train = xs[idx_train]
            nSamp = loader.nSamp
            
        else:
            
            metx_train, labels, idx = loader.load_csv(path_to_metadata, farm=farm_train)
            xs_train, _, _ = loader.load_csv(path_to_xs, idx=idx)
            metx_test, labels, idx = loader.load_csv(path_to_metadata, farm=farm_test)
            xs_test, _, _ = loader.load_csv(path_to_xs, idx=idx)
            idx_train = np.arange(len(metx_train))
            idx_test = np.arange(len(metx_test))
            idx_val = []
            metz_train, _, _ = loader.preprocess_metadata(metx_train, idx_train, [], idx_val)
            _, metz_test, _ = loader.preprocess_metadata(metx_test, idx_test, idx_test, idx_val)
            xs_val = [] 
            metz_val = []
            nSamp = np.nan

        [_, nMet] = metz_train.shape
        
        # INITIALIZING DICTS AND OBJECTS
        param_dict = {"nK": nK, "alph": alph, "etaZ": 3e-5, "etaT": 3e-5, "etaC": 3e-5, "nPC": nK - nL, "nL": nL,
                      "nMet": nMet, "nD": xs_test.shape[1], "nSamp": nSamp, "mu_met": loader.mu_met,
                      "sg_met": loader.sg_met, "runID": runID, "extra_spec": extraspec}
        if param_dict["extra_spec"] is None:
            param_dict.pop("extra_spec")

        eval_dict = {"e1": np.nan, "e2": np.nan, "KL_final": np.nan, "KL_train": [0], "KL_val": [], "spearman": [], "pearson":  []}

        grad_dict = {"grz": np.nan, "grthet": np.nan, "grc": np.nan}

        data_dict = {"xs_train": xs_train, "xs_test": xs_test, "xs_val": xs_val, "Q": np.nan,
                     "metz_train": metz_train, "metz_test": metz_test, "metz_val": metz_val, "C": np.nan, "Z": None,
                     "Zcon": np.nan, "Zfr": np.nan,
                     "thet": np.random.random(size=(param_dict["nPC"] + param_dict["nL"], param_dict["nD"])),
                     "metx_labels": labels, "idx_train": idx_train, "idx_test": idx_test, "idx_val": idx_val}
        

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
        print("pearson", PredObj.pearson)
        
        if farm_train != farm_test:
            break
        
def PDR_alph_range(paths_to_data, writer, extraspec, nK, nL, alph_arr, num_runs, samp_freq, stop_search, max_iter, 
                   farm_train=None, farm_test=None):
    for alph in alph_arr:
        alph = float('%.3f' % alph)
        PDR_algo(paths_to_data, writer, extraspec, nK, nL, alph, num_runs, samp_freq, stop_search, max_iter,
                 farm_train, farm_test)
        
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
    print(p)
    os.makedirs(p, exist_ok=True)
    
    return path_to_hdf

def step_arr(start, stop, step):
    arr = np.linspace(start, stop, int((stop - start)/step + 1))
    for i in range(len(arr)):
        arr[i] = float('%.3f' % arr[i])
    return arr

# alph_arr = step_arr(0.05, 0.30, 0.05)
alph_arr = [0.05]
path_to_metadata = "/blue/pdixit/lukasherron/parallel_dim_reduction/data/datasets/bovine_metadata.csv"
path_to_xs = "/blue/pdixit/lukasherron/parallel_dim_reduction/data/datasets/bovine_microbiome.csv"
paths_to_data = [path_to_metadata, path_to_xs]

nL = 2
nK = 12
num_runs = 10
farm_train = 2
farm_test = 3

naming_dict = {"model_type": 'testrun',
               "extra_spec": "farm_train="+ str(farm_train) + "_farm_test=" + str(farm_test),
               "nK": "nK=" + str(nK),
               "nL": "nL=" + str(nL)}

path_to_hdf = init_file_struct(path_to_metadata, naming_dict)

writer = DataWriter(path_to_hdf, mode="w")

PDR_alph_range(paths_to_data, writer, naming_dict["extra_spec"], nK, nL, alph_arr, num_runs, 5000, 55000, 100000,
               farm_train=farm_train, farm_test=farm_test)
writer.close()