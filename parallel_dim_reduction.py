#!/usr/bin/env python
from sklearn.decomposition import PCA
from scipy.io import loadmat
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter1d
import os
import glob
from tables import *
import sys

#------------------------------PYTABLES UTILITIES--------------------------------
def get_dtype(data):
    """Given a dict, generate a nested numpy dtype"""

    if sys.version.startswith('3'):
        unicode = str
    fields = []
    for (key, value) in data.items():
        # make strings go to the next 64 character boundary
        # pytables requires an 8 character boundary
        if isinstance(value, unicode):
            value += u' ' * (64 - (len(value) % 64))
            # pytables does not support unicode
            if isinstance(value, unicode):
                value = value.encode('utf-8')
        elif isinstance(value, str):
            value += ' ' * (64 - (len(value) % 64))

        if isinstance(value, dict):
            fields.append((key, get_dtype(value)))
        else:
            value = np.array(value)
            fields.append((key, '%s%s' % (value.shape, value.dtype)))
    return np.dtype(fields)

def unpack(row, base, data):
    """ Unpacks th entries of a dict. """

    for (key, value) in data.items():
        new = base + key
        if isinstance(value, dict):
            unpack(row, new + '/', value)
        else:
            row[new] = value

def add_row(tbl, data):
    """Add a new row to a table based on the contents of a dict."""

    row = tbl.row
    for (key, value) in data.items():
        if isinstance(value, dict):
            unpack(row, key + '/', value)
        else:
            row[key] = value
    row.append()
    tbl.flush()

def navigate_to_table(h5file, ArbObj, dic):
    """ Navigates to a table based on a specified path and the reference string of a dict. If the 
    groups along the path to the table do not exists or the table does not exist it is created.

    In the future this function will be changed so that the path is specified by **kwargs."""

    path_deconstruct = [dic[key] for key in dic.keys()]
    path_construct = os.path.join(*path_deconstruct)
    group_exists, table_exists = False, False
    for group in h5file.walk_groups():
        if path_construct == group._v_pathname:
            group_exists = True
            break

    if group_exists is False:
        group_path = "/"
        for group_name in path_deconstruct:
            try:
                group = h5file.create_group(group_path, group_name, createparents=True)
            except:
                pass
            group_path = os.path.join(group_path, group_name)

    for table in h5file.list_nodes(group):
        if ArbObj.obj_type == table._v_name:
            table_exists = True
            break

    if table_exists is False:
        dtype = get_dtype(ArbObj.__dict__)
        table = h5file.create_table(group, ArbObj.obj_type, dtype)

    return table


#-------------------------------SYSTEM UTILITIES----------------------------------
def get_str_from_name(filename, param_name):
    _, filename = pop_path(filename)
    i = filename.find(param_name)
    j = len(param_name)
    temp = filename[i+j+1:]
    k = temp.find('_')
    param_value = temp[0:k]
    return param_value

def pop_path(filename):
    i = 0
    path = ''
    while i != -1:
        i = filename.find("/")
        path += filename[:i+1]
        filename = filename[i+1:]
    return path, filename

def subdirs(path):
    for entry in os.scandir(path):
        if not entry.name.startswith('.') and entry.is_dir():
            yield entry.name

def find_files(path):
    result = [y for x in os.walk(path) for y in glob.glob(os.path.join(x[0], '*.h5'))]     
    return result

def consolidate_data(path_to_root, path_to_h5):

    files = list(find_files(path_to_root))
    h5fw = open_file(path_to_h5, mode='w')
    for h5name in files:
        h5fr = open_file(h5name, mode='r') 
        h5fr.root._f_copy_children(h5fw.root, recursive=True)
        h5fw.flush()
        h5fr.close()
    h5fw.close()
    
#------------------------------------------------------------------------------


def KL_divergence(arr_1, arr_2):
    """ Calculates the KL Divergence between arr_1 and arr_2. Here arr_1 is the 
    test data and arr_2 is the corresponding predicted data. """

    temp = np.ma.log(np.divide(arr_1, arr_2, where=arr_2 != 0).astype('float64'))
    temp = temp.filled(0)
    temp[np.isnan(temp)] = 0
    KL = np.nansum(np.nansum(np.multiply(arr_1, temp)))

    return KL

class ObjFromDict(object):
    """ Maps key/value pairs of a dict to attributes of an object, with identifier 
    string obj_ref"""
    def __init__(self, obj_dict, obj_ref):
        for key in list(obj_dict.keys()):
            setattr(self, key, obj_dict[key])
        setattr(self, "obj_keys", list(obj_dict.keys()))
        setattr(self, "obj_type", obj_ref)

class Parallel_Dimensionality_Reduction(object):

    def __init__(self, ParamObj, DataObj, EvalObj, GradObj, PredObj=None):
        self.DataObj = DataObj
        self.ParamObj = ParamObj
        self.EvalObj = EvalObj
        self.GradObj = GradObj
        self.PredObj = PredObj

    def eval_Q(self):
        """ Evaluates TMI Predictions from Z and thet. Includes making predictions
        and normalization. """

        self.DataObj.Q = np.exp(-self.DataObj.Z @ self.DataObj.thet)
        self.DataObj.Q[np.isnan(self.DataObj.Q)] = 0
        norms = self.DataObj.Q.sum(1)
        self.DataObj.Q = np.divide(self.DataObj.Q.T, norms, where=norms != 0).T

    def PCA_TMI_gradient_update(self):
        """ Updates the gradients of the relevant variables assuming that PCA reduction
        is performed"""

        delt = self.DataObj.xs_train - self.DataObj.Q
        self.DataObj.Zcon = self.DataObj.Z[:, 0:self.ParamObj.nPC]
        self.GradObj.thet = (1 - self.ParamObj.alph) * self.DataObj.Z.T @ delt
        self.GradObj.grz = (1 - self.ParamObj.alph) * delt @ self.DataObj.thet.T
        # gradient for free components

        grz_con = -2 * self.ParamObj.alph * (self.DataObj.metz_train - self.DataObj.Zcon @ \
                                             self.DataObj.C) @ self.DataObj.C.T
        self.GradObj.grz[:, 0:self.ParamObj.nPC] += grz_con
        self.GradObj.grc = -2 * self.ParamObj.alph * self.DataObj.Zcon.T @ \
        (self.DataObj.metz_train - self.DataObj.Zcon @ self.DataObj.C)
        self.DataObj.Z -= self.ParamObj.etaZ * self.GradObj.grz
        self.DataObj.thet -= self.ParamObj.etaT * self.GradObj.thet
        self.DataObj.C -= self.ParamObj.etaC * self.GradObj.grc

    def eval_training_performance(self):
        """ Evaluates the KL divergence and relative errors between the training model. """
        
        self.EvalObj.e1 = np.linalg.norm(self.DataObj.metz_train - self.DataObj.Zcon @ \
                                         self.DataObj.C)
        self.EvalObj.e1 = self.ParamObj.alph * self.EvalObj.e1 ** 2

        self.EvalObj.KL_train = KL_divergence(self.DataObj.xs_train, \
                                              self.DataObj.Q)/len(self.DataObj.idx_train)
        self.EvalObj.e2 = (1 - self.ParamObj.alph) * self.EvalObj.KL_train

    def step(self):
        """ One iteration of the gradient descent algorithm."""

        self.eval_Q()
        self.PCA_TMI_gradient_update()

    def gradient_descent(self, path, samp_freq, stop_search, maxiter, validate=True, verbose=True):
        """ Complete gradient descent algorithm with the option to validate the training data on 
        some validation set calculated with the validation data from the split_data(...) function 
        under the DataLoader class.
        
        In the future a general early stopping mechanism will be added."""
        
        i = 0
        check_stop = False

        while i < maxiter:
            self.step()
            i += 1
            if i % samp_freq == 0:
                self.eval_training_performance()
                if validate:
                    PFM = Predict_From_Model(PredObj, DataObj, ParamObj)
                    PFM.predict_microbiome(validate=validate)
                    self.EvalObj.KL_val.append(PFM.PredObj.KL)
                    KL_1 = np.array(self.EvalObj.KL_val)
                    KL_2 = np.array(self.EvalObj.KL_train)
                    KL_tot = (KL_1)[int((stop_search - samp_freq) / samp_freq):]

        self.eval_training_performance()
        if validate:
            PFM = Predict_From_Model(PredObj, DataObj, ParamObj)
            PFM.predict_microbiome(validate=validate)
            self.EvalObj.KL_val.append(PFM.PredObj.KL)

    def finalize(self):
        """ Returns objects. Always call finalize to make the changes made to objects from the 
        PDR_reduction class persistent."""
        
        return self.DataObj, self.EvalObj, self.GradObj


class PCA_reduction(object):

    def __init__(self, ParamObj, DataObj, EvalObj):
        self.DataObj = DataObj
        self.ParamObj = ParamObj
        self.EvalObj = EvalObj

    def dim_reduction(self, guess=True):
        """ Performs PCA reduction on the metadata passed to DataObj."""

        [nSamp, _] = self.DataObj.xs_train.shape
        pca = PCA(n_components=self.ParamObj.nPC)
        self.DataObj.Zcon = pca.fit_transform(self.DataObj.metz_train)
        [self.DataObj.C, _] = [pca.components_, pca.explained_variance_]
        self.DataObj.C = self.DataObj.C.T
        self.DataObj.C = self.DataObj.C[:self.ParamObj.nPC, :]

        if guess:
            self.DataObj.C = np.random.normal(size=(self.ParamObj.nPC, self.ParamObj.nMet))
            Zfr = np.random.random(size=(nSamp, self.ParamObj.nL))
            self.DataObj.Z = np.hstack((self.DataObj.Zcon, Zfr))

    def finalize(self):
        """ Returns objects. Always call finalize to make the changes made to objects from the 
        PCA_reduction class persistent."""
        
        return self.DataObj, self.EvalObj
    
class DataWriter(object):

    def __init__(self, path_to_h5, mode):
        self.h5file = open_file(path_to_h5, mode=mode)
        
    def write_obj(self, ParamObj, ArbObj):
        """ Writes an object to a table specified by the obj_type reference attribute of the object."""
        try:
            dic = {"extra_spec": str(ParamObj.extra_spec),
                   "nK": "nK=" + str(ParamObj.nK),
                   "nL": "nL=" + str(ParamObj.nL),
                   "alph": "alph=" + str(ParamObj.alph)}
        except:
            dic = {"nK": "nK=" + str(ParamObj.nK),
                   "nL": "nL=" + str(ParamObj.nL),
                   "alph": "alph=" + str(ParamObj.alph)}
            
        table = navigate_to_table(self.h5file, ArbObj, dic)
        dtype = get_dtype(ArbObj.__dict__)
        add_row(table, ArbObj.__dict__)
        
    def close(self):
        """ Saves and closes the file which the DataWriter object modifies."""
        
        self.h5file.flush()
        self.h5file.close()
        

class DataLoader(object):

    def __init__(self, path=None, mode="r"):
        self.mat = None
        self.labels = None
        self.nMet = None
        self.mat_keys = None
        self.nSamp = None
        self.mu_met = None
        self.sg_met = None
        if path is not None and mode is not None:
            self.h5file = open_file(path, mode=mode)
            self.path = path            

    def load_data(self, path):
        """Loads the dataset located at the specified path. Currently the dataset must be in .mat 
        format.
        
        In the future the function will be able to load excel files or accept pandas datasets and numpy 
        arrays/labels."""
        
        mat = loadmat(path)
        self.mat_keys = mat.keys()
        self.mat = mat
        xs = mat['xs'].toarray()
        metx = mat["metx"]
        labels = [mat["metnames"][0][i][0] for i in range(len(mat["metnames"][0]))]
        [nSamp, nMet] = metx.shape

        self.nMet = nMet
        self.nSamp = nSamp

        return xs, metx, labels

    def split_data(self, test_size, val_size):
        """ Splits data into testing, training, and validation sets. test_size and val_size are the 
        proporitons of the original dataset that will be included in the test and validation set.
        
        This function is deprecated and the handling of empty validation sets will be handled more 
        elegantly in the future."""
        indices = list(range(self.nSamp))
        if val_size == 0:
            train_indices, test_indices = train_test_split(indices, test_size=test_size)
            val_indices = [[]]
        else:
            train_indices, test_and_val_indices = train_test_split(indices, test_size=test_size \
                                                                   + val_size)
            test_indices, val_indices = train_test_split(test_and_val_indices, \
                                                         test_size=val_size/(test_size + val_size))

        return train_indices, test_indices, val_indices


    def preprocess_metadata(self, metx, train_idx, test_idx, val_idx):
        """ Standardizes metadata. Demeaning and dividing by standard deviation."""

        metx_train = metx[train_idx]
        metx_test = metx[test_idx]
        metx_val = metx[val_idx]
        mu_met = np.mean(metx_train, axis=0)
        sg_met = np.diagonal(np.cov(metx_train.T))
        metz_train = np.divide((metx_train - mu_met), sg_met)
        metz_test = np.divide((metx_test - mu_met), sg_met)
        metz_val = np.divide((metx_val - mu_met), sg_met)

        self.mu_met = mu_met
        self.sg_met = sg_met

        return metz_train, metz_test, metz_val
    
    def load_obj(self, nK, nL, alph, obj_type, runIDs=all, entries=all):
        """ Loads an object from an h5file. nK, nL, and alph specify the parameters of the model. 
        The table in the h5file is specified by obj_type which can be "param", "data", "eval", or 
        "pred". RunIDs handles the case of multiple model runs while entries enables a single entry 
        of the table to be returned. The retured data takes the form of a dict, where the keys are 
        the column names of the entries in the table and the values are the corresponding entries.  
        
        In the future this definition will be able to handle a consolidated dataset. Currently 
        params nK and nL specify directories, but the ability to contain nK, nL, alph in a single 
        h5file will be added.
        """
        
        if self.path.find("consolidated") == -1:
            path_construct = os.path.join("/", "alph=" + str(alph), obj_type)
            data_table = self.h5file.get_node(path_construct)
            data = {}
            labels = data_table.colnames
            if runIDs is not all:
                i = 0
                for runID in runIDs:
                    i += 1
                    if entries is not all:
                        for entry in entries:
                            label_idx = labels.index(entry)
                            if i == 1:
                                data[labels[label_idx]] = []
                            data[labels[label_idx]].append(data_table[runID][label_idx])
                    else:
                        values = [x for x in data_table[runID]]
                        for idx, label in enumerate(labels):
                            data[label] = values[idx]
            else:
                if entries is not all:
                    for entry in entries:
                        label_idx = labels.index(entry)
                        data[labels[label_idx]] = [x[label_idx] for x in data_table.iterrows()]
                else:
                    for idx, label in enumerate(labels):
                        data[label] = [x[idx] for x in data_table.iterrows()]
        return data
        

    def load_from_params(self, nK, nL, alph, entry, data_type):
        """ This function is deprecated and must be changed to be compatible with h5files. """

        pass

    def close(self):
        """ Closes the h5file that the DataLoader reads. """
        self.h5file.close()

class Predict_From_Model(object):

    def __init__(self):

        pred_dict = {
            "Q": np.nan,
            "Z": np.nan,
            "Zfr": np.nan,
            "C": np.nan,
            "KL": np.nan,
        }

        self.ParamObj = None
        self.EvalObj = None
        self.DataObj = None
        self.PredObj = ObjFromDict(pred_dict, "pred")

    def load_model(self, path_to_h5file, nK, nL, alph, runID):
        """Loads a model from a h5file. """
        loader = DataLoader(path_to_h5file, mode="r")
        model_param_dict= loader.load_obj(nK, nL, alph, "param", runIDs=runID, entries=all)
        model_eval_dict = loader.load_obj(nK, nL, alph, "eval", runIDs=runID, entries=all)
        model_data_dict = loader.load_obj(nK, nL, alph, "data", runIDs=runID, entries=all)
        self.ParamObj = ObjFromDict(model_param_dict, "param")
        self.EvalObj = ObjFromDict(model_eval_dict, "eval")
        self.DataObj = ObjFromDict(model_data_dict, "data")
        
        loader.close()
        
    def inherit_model(self, ParamObj, DataObj, EvalObj):
        """ Passes the objects from a model to a PredictFromModel object."""
        
        self.ParamObj = ParamObj
        self.DataObj = DataObj
        self.EvalObj = EvalObj
    
    def estimate_free_components(self, Z_train, Z_pred, validate):
        """ Estimates the free components that are assumed to be gaussian distributes. The assmuned gaussian
        is conditionalized over the training data and the free components are sampled from the conditionalized 
        distribution. """

        nPC = self.ParamObj.nPC
        nL = self.ParamObj.nL
        if validate is False:
            idx_test = self.DataObj.idx_test
        if validate is True:
            idx_test = self.DataObj.idx_val

        muZ = np.mean(Z_train, axis=0).T
        muZ = muZ[:, np.newaxis]
        z2 = Z_pred[:, :nPC]  
        mu2 = muZ[:nPC]
        mu1 = muZ[nPC:]
        covZ = np.cov(Z_train.T) 
        cov11 = covZ[nPC:, nPC:]
        cov12 = covZ[nPC:, :nPC]
        cov21 = cov12.T
        cov22 = covZ[:nPC, :nPC]

        sigbar = cov11 - cov12 @ np.linalg.inv(cov22) @ cov21
        nTest = len(idx_test)
        Zfr = np.zeros((nTest, nL))
        for i in range(nTest):
            z2_train = Z_pred[i, :nPC]
            z2_train = z2_train[:, np.newaxis]

            mubar = mu1 + cov12 @ np.linalg.inv(cov22) @ (mu2 - z2_train)
            Zfr[i, :] = np.random.multivariate_normal(mubar.ravel(), sigbar)

        Z = np.hstack((Z_pred, Zfr))

        return Zfr, Z

    def predict_microbiome(self, validate=False):
        """ Predicts the microbiome from corresponding metadata samples using the learned C, Z, and thet. """

        if validate is False:
            test_xs = self.DataObj.xs_test
            metz_test = self.DataObj.metz_test
            idx_test = self.DataObj.idx_test

        if validate is True:
            test_xs = self.DataObj.xs_val
            metz_test = self.DataObj.metz_val
            idx_test = self.DataObj.idx_val

        C = self.DataObj.C
        Z_train = self.DataObj.Z
        thet =self.DataObj.thet

        Z_pred = metz_test @ np.linalg.pinv(self.DataObj.C)
        if self.ParamObj.nL != 0:
            self.PredObj.Zfr, self.PredObj.Z = self.estimate_free_components(self.DataObj.Z, Z_pred, validate)
            Z = self.PredObj.Z
        else:
            Z = Z_pred

        self.PredObj.Q = np.exp(-Z @ self.DataObj.thet)
        self.PredObj.Q[np.isnan(self.PredObj.Q)] = 0
        norms = self.PredObj.Q.sum(1)
        self.PredObj.Q = np.divide(self.PredObj.Q.T, norms, where=norms != 0).T
        self.PredObj.KL = KL_divergence(test_xs, self.PredObj.Q)/len(idx_test)
        
    def predict_metadata(self, validate=False):
        """ Predicts the metadata from corresponding microbiome samples using C, Z, and thet."""
        
        pred_dict = self.pred_dict
        self.model_param_dict = model_param_dict
        self.model_data_dict = model_data_dict

        if validate is False:
            test_xs = self.DataObj.xs_test
            metz_test = self.DataObj.metz_test
            idx_test = self.DataObj.idx_test

        if validate is True:
            test_xs = self.DataObj.xs_val
            metz_test = self.DataObj.metz_val
            idx_test = self.DataObj.idx_val

        nPC = self.ParamObj.nPC
        
        Z_pred = -np.log(self.DataObj.Q) @ np.linalg.pinv(self.DataObj.thet)
        Z_restricted = Z_pred[0:nPC, :]
        self.PredObj.metz = Z_restricted @ self.DataObj.C        
        self.PredObj.KL = KL_divergence(metz_test, metz_pred)

    def finalize(self):
        """ Returns objects. Always call finalize to make the changes made to objects from the 
        PredictFromModel class persistent."""
        return self.PredObj
        