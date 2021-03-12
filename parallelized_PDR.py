#!/usr/bin/bash/env python
from parallel_dim_reduction import *
import numpy as np
import matplotlib.pyplot as plt
from plots_and_figs import *


def PDR_algo(path_to_data, model_type, further_spec, nK, nL, alph, samp_freq, stop_search, max_iter):

    for num in range(20):

        path_to_data_dir = "/blue/pdixit/lukasherron/parallel_dim_reduction/data/"
        dataset_name = create_filetree(path_to_data_dir, path_to_data, model_type, further_spec=further_spec)
        fs = str(further_spec)
        if fs == "None":
            fs = ""
        path_stem = "/blue/pdixit/lukasherron/parallel_dim_reduction/data/" + str(dataset_name) + "/" + str(model_type) + "/" + fs + "/"
        
        # LOADING DATA
        loader = DataLoader()
        xs, metx, labels = loader.load_data(path_to_data)
        idx_train, idx_test, idx_val = loader.split_data(test_size=0.15, val_size=0.15)
        metz_train, metz_test, metz_val = loader.preprocess_metadata(metx, idx_train, idx_test, idx_val)

        # INITIALIZING DICTS AND OBJECTS
        param_dict = {"nK": nK, "alph": alph, "etaZ": 3e-5, "etaT": 3e-5, "etaC": 1e-5, "nPC": nK - nL, "nL": nL,
                      "nMet": loader.nMet, "nD": xs.shape[1], "nSamp": loader.nSamp, "mu_met": loader.mu_met,
                      "sg_met": loader.sg_met}

        eval_dict = dict(e1=None, e2=None, KL_final=None, KL_train=[], KL_val=[])

        grad_dict = dict(grz=None, grthet=None, grc=None)

        data_dict = {"xs_train": xs[idx_train], "xs_test": xs[idx_test], "xs_val": xs[idx_val], "Q": None,
                     "metz_train": metz_train, "metz_test": metz_test, "metz_val": metz_val, "C": None, "Z": None,
                     "Zcon": None, "Zfr": None,
                     "thet": np.random.random(size=(param_dict["nPC"] + param_dict["nL"], param_dict["nD"])),
                     "labels": labels, "idx_train": idx_train, "idx_test": idx_test, "idx_val": idx_val}

        pointer_to_model = path_stem + "/models/nK=" + str(nK) + "/nL=" + str(nL) + "/"
        filename_model = "model_nK=" + str(nK) + "_nL=" + str(nL) + "_alph=" + str(alph) + "_num=" + str(num) + "_.xlsx"
        path_to_model = pointer_to_model + filename_model
        writer = DataWriter()
        writer.create_model_workbook(path_to_model, data_dict)

    # TRAINING
        # PCA is a good initial guess for PDR
        PCA_red = PCA_reduction(param_dict, data_dict, eval_dict)
        PCA_red.dim_reduction(guess=True)
        data_dict, eval_dict = PCA_red.finalize()

        # Parallel Dimensionality Reduction
        PDR = Parallel_Dimensionality_Reduction(param_dict, data_dict, eval_dict, grad_dict)
        PDR.gradient_descent(path_to_model, samp_freq, stop_search, max_iter, validate=True)
        print("gradient descent complete")
        PDR.eval_training_performance()
        data_dict, eval_dict, grad_dict = PDR.finalize()  # Model is saved

    # TESTING PHASE
        pointer_to_pred = path_stem + "/predictions/nK=" + str(nK) + "/nL=" + str(nL) + "/"
        filename_pred = "model_nK=" + str(nK) + "_nL=" + str(nL) + "_alph=" + str(alph) + "_num=" + str(num) + "_pred.xlsx"
        path_to_pred = pointer_to_pred + filename_pred

        # Loading model
        model_param_dict, model_eval_dict, model_data_dict = loader.load_model(path_to_model)

        # Making predictions
        PFM = Predict_From_Model()
        PFM.load_model(path_to_model)
        PFM.predict_microbiome(model_param_dict, model_data_dict)
        pred_dict = PFM.finalize()
        
        Q = pred_dict["Q"].ravel()
        x = pred_dict["xs_test"].ravel()

#         fig, ax = plt.subplots(figsize=(8,8))
#         plt.scatter(np.log10(x), np.log10(Q), s=3)

#         ax.tick_params(direction='in', length=6, width=2, top=True, right=True, which='major')
#         ax.tick_params(direction='in', length=3, width=0.5, top=True, right=True, which='minor')

#         ax.set_xlabel("test xs", size=16)
#         ax.set_ylabel("test Q", size=16)
#         for axis in ['top', 'bottom', 'left', 'right']:
#             ax.spines[axis].set_linewidth(2)
#         for tick in ax.xaxis.get_ticklabels():
#             tick.set_fontsize(16)
#             tick.set_fontname('serif')
#         for tick in ax.yaxis.get_ticklabels():
#             tick.set_fontsize(16)
#             tick.set_fontname('serif')
#         plt.legend(loc = "lower right", prop={'size': 12}, framealpha=1)
#         fig_title = "Prediction Scatter (nK=" + str(nK) + ", nL=" + str(nL) + ", alph=" + str(alph) + ") (log)"

#         plt.title(fig_title, size=18);
#         plt.xlim(-5, 0)
#         plt.ylim(-5, 0)
#         pointer_to_fig = plots_and_figs + "/plots_and_figs/nK=" + str(nK) + "/nL=" + str(nL) + "/"
#         figname = "prediction_scatterplot_nK=" + str(nK) + "_nL=" + str(nL) + "_alph=" + str(alph) + ".jpg"
#         plt.savefig(pointer_to_fig + figname)
#         plt.close()

        # Saving predictions
        writer.create_pred_workbook(path_to_pred, pred_dict)
        writer.write_pred_workbook(path_to_pred, pred_dict)


def PDR_alph_range(path_to_data, model_type, further_spec, nK, nL, alph_arr, samp_freq, stop_search, max_iter):
    for alph in alph_arr:
        alph = float('%.3f' % alph)
        PDR_algo(path_to_data, model_type, further_spec, nK, nL, alph, samp_freq, stop_search, max_iter)

alph_max = 0.3
alph_div = 0.01
alph_arr = np.arange(0, int(alph_max/alph_div) + 1) * alph_div
path_to_data = "/blue/pdixit/lukasherron/parallel_dim_reduction/data/datasets/bovine_data.mat"
model_type = "distribution_runs"
further_spec=None

num_L_components = 4
num_K_components = 28

PDR_alph_range(path_to_data, model_type, further_spec, num_K_components, num_L_components, alph_arr, 100, 1000, 20000)