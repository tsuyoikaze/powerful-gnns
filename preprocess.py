import sys
import pandas as pd
import numpy as np
import os
import re
import json
from shutil import copyfile

metadata_labels = ['ImageNumber','ObjectNumber','Metadata_FileLocation','Metadata_Frame','Metadata_Series','Metadata_cancer_class','Metadata_cancer_type','Metadata_id','Metadata_patient','Metadata_patient.1']
metadata_length = 10

features_not_to_remove= ['cell_Intensity_MassDisplacement_E', 'nuc_Intensity_MassDisplacement_E', 'cell_Location_CenterMassIntensity_X_E', 'nuc_Location_CenterMassIntensity_X_E', 'cell_Location_CenterMassIntensity_Y_E', 'nuc_Location_CenterMassIntensity_Y_E', 'cell_Location_CenterMassIntensity_Z_E', 'nuc_Location_CenterMassIntensity_Z_E', 'cell_Intensity_UpperQuartileIntensity_E', 'nuc_Intensity_UpperQuartileIntensity_E', 'cell_Intensity_UpperQuartileIntensity_H', 'nuc_Intensity_UpperQuartileIntensity_H', 'cell_Location_CenterMassIntensity_X_E', 'nuc_Location_CenterMassIntensity_X_E', 'cell_Location_CenterMassIntensity_X_H', 'nuc_Location_CenterMassIntensity_X_H', 'cell_Location_CenterMassIntensity_Y_E', 'nuc_Location_CenterMassIntensity_Y_E', 'cell_Location_CenterMassIntensity_Y_H', 'nuc_Location_CenterMassIntensity_Y_H', 'cell_Location_CenterMassIntensity_Z_E', 'nuc_Location_CenterMassIntensity_Z_E', 'cell_Location_CenterMassIntensity_Z_H', 'nuc_Location_CenterMassIntensity_Z_H']
features_to_remove = ['cell_AreaShape_Center_Z', 'nuc_AreaShape_Center_Z', 'cell_AreaShape_Orientation', 'nuc_AreaShape_Orientation', 'cell_Location_CenterMassIntensity_X_E', 'nuc_Location_CenterMassIntensity_X_E', 'cell_Location_CenterMassIntensity_X_H', 'nuc_Location_CenterMassIntensity_X_H', 'cell_Location_CenterMassIntensity_Y_E', 'nuc_Location_CenterMassIntensity_Y_E', 'cell_Location_CenterMassIntensity_Y_H', 'nuc_Location_CenterMassIntensity_Y_H', 'cell_Location_CenterMassIntensity_Z_E', 'nuc_Location_CenterMassIntensity_Z_E', 'cell_Location_CenterMassIntensity_Z_H', 'nuc_Location_CenterMassIntensity_Z_H', 'cell_Location_Center_X', 'nuc_Location_Center_X', 'cell_Location_Center_Y', 'nuc_Location_Center_Y', 'cell_Location_Center_Z', 'nuc_Location_Center_Z', 'cell_Location_MaxIntensity_X_E', 'nuc_Location_MaxIntensity_X_E', 'cell_Location_MaxIntensity_X_H', 'nuc_Location_MaxIntensity_X_H', 'cell_Location_MaxIntensity_Y_E', 'nuc_Location_MaxIntensity_Y_E', 'cell_Location_MaxIntensity_Y_H', 'nuc_Location_MaxIntensity_Y_H', 'cell_Location_MaxIntensity_Z_E', 'nuc_Location_MaxIntensity_Z_E', 'cell_Location_MaxIntensity_Z_H', 'nuc_Location_MaxIntensity_Z_H', 'cell_Number_Object_Number', 'nuc_Number_Object_Number', 'cell_Parent_PreNucleus', 'nuc_Parent_PreNucleus', 'cell_ImageNumber', 'nuc_ImageNumber', 'cell_ObjectNumber', 'nuc_ObjectNumber', 'cell_Metadata_FileLocation', 'nuc_Metadata_FileLocation', 'cell_Metadata_Frame', 'nuc_Metadata_Frame', 'cell_Metadata_Series', 'nuc_Metadata_Series', 'cell_Metadata_cancer_class', 'nuc_Metadata_cancer_class', 'cell_Metadata_cancer_class.1', 'nuc_Metadata_cancer_class.1', 'cell_Metadata_cancer_type', 'nuc_Metadata_cancer_type', 'cell_Metadata_cancer_type.1', 'nuc_Metadata_cancer_type.1', 'cell_Metadata_id', 'nuc_Metadata_id', 'cell_Metadata_patient', 'nuc_Metadata_patient', 'cell_Metadata_patient.1', 'nuc_Metadata_patient.1']
graph_features_to_remove = ['cell_AreaShape_Center_X', 'nuc_AreaShape_Center_X', 'cell_AreaShape_Center_Y', 'nuc_AreaShape_Center_Y']



def to_type(csv, label, dtype):
    '''
    Convert a column to another datatype

    Parameters:
        csv:   the pandas DataFrame
        label: the label of a column
        dtype: the data type to be converted

    Returns:
        The converted dataframe
    '''
    csv[label] = csv[label].astype(dtype)
    return csv

def check_nan(csv, label):
    '''
    Check if column contains NaN value

    Parameters: 
        csv:   the pandas DataFrame
        label: the label of a column

    Returns:
        True if it contains NaN value; False otherwise
    '''
    return csv[label].astype(float).isnull().any()

def remove_nan(csv, label):
    '''
    Remove column if it contains NaN value

    Parameters: 
        csv:   the pandas DataFrame
        label: the label of a column

    Returns:
        The chopped DataFrame
    '''
    if check_nan(csv, label):
        return csv.drop(columns=[label])
    return csv 

def split_data(fpaths, labels, n, p):
    '''
    Split the data into multiple datasets

    Parameters: 
        fpaths: the list of file paths to be splitted
        labels: the numpy array of the labels corresponding to the paths
        n:      the number of sets to be splitted
        p:      the numpy array of percentage of samples in each set

    Returns:
        List of 2-tuples. Each 2-tuple represents the samples and labels in numpy arrays
    '''
    matrix = np.array(fpaths)
    idx = np.arange(matrix.shape[0])
    np.random.shuffle(idx)
    matrix_shuffled = matrix[idx]
    labels_shuffled = labels[idx]
    set_length = np.round(p * matrix.shape[0]).astype(int)
    while np.sum(set_length) > matrix.shape[0]:
        set_length[set_length.argmax()] -= 1
    while np.sum(set_length) < matrix.shape[0]:
        set_length[set_length.argmin()] += 1
    curr_idx = 0
    res = []
    for i in range(n):
        res.append((matrix_shuffled[curr_idx:curr_idx+set_length[i]], labels_shuffled[curr_idx:curr_idx+set_length[i]]))
        curr_idx += set_length[i]
    return res
    
def binning(fpaths, labels):
    res = dict()
    for i in range(len(fpaths)):
        if labels[i] not in res:
            res[labels[i]] = np.array([])
        res[labels[i]] = np.append(res[labels[i]], [fpaths[i]])
    return res   

def get_labels(fpaths, depth = 'full'):
    '''
    Gives the label (cancer type, etc) base on the list of samples as filenames

    Parameters: 
        fpaths: the list of file paths to be splitted

    Returns
        Labels of the labels as numpy array of int and a dictionary mapping from label string to int
    '''
    if depth == 'full':
        depth_flag = True
    else:
        depth_flag = False
    res = []
    last_idx = 0
    labels_dict = dict()
    for fpath in fpaths:
        match = re.match(".*(\w+)\/(\w+)\/([^/\\\n\r]+)(?:\/.*\.csv)?", fpath)
        label = match.group(1) + ('_' + match.group(2) if depth == 'full' else '')
        if label not in labels_dict:
            labels_dict[label] = last_idx
            last_idx += 1
        res.append(labels_dict[label])
    return np.array(res), labels_dict

def write_list(l, fname):
    with open(fname, 'w') as f:
        for i in l:
            f.write(str(i) + '\n')

def write_dict(d, fname):
    with open(fname, 'w') as f:
        f.write(json.dumps(d))

def read_list(fname):
    d = []
    with open(fname) as f:
        for l in f:
            d.append(l.strip())
    return np.array(d)

def read_dict(fname):
    res = None
    with open(fname) as f:
        res = json.load(f)
    return res

def process_sample(dir_path):

    # load csv
    cell_csv = pd.read_csv(os.path.join(dir_path, 'Cell.csv')).add_prefix('cell_')
    nucleus_csv = pd.read_csv(os.path.join(dir_path, 'Nucleus.csv')).add_prefix('nuc_')

    # combine and filter out small parts
    combined_csv = pd.concat([cell_csv, nucleus_csv], axis = 1)
    combined_csv = combined_csv[combined_csv['cell_AreaShape_Area'] >= 10]
    combined_csv = combined_csv[combined_csv['nuc_AreaShape_Area'] >= 10]

    # assign NaN intensities to 0
    combined_csv[features_not_to_remove].fillna(value = 0.0)

    groups = combined_csv.groupby('cell_Metadata_id')
    features = []
    graphs = []
    ids = []
    for name, group in groups:
        ids.append(name)
        group = group.sort_values(by = ['cell_ObjectNumber'])
        for feature_to_remove in features_to_remove:
            if feature_to_remove in group.columns:
                group = group.drop(columns=feature_to_remove)
        
        # remove any NaN cell
        group = group.dropna()
        graph = group[['cell_AreaShape_Center_X', 'cell_AreaShape_Center_Y', 'nuc_AreaShape_Center_X', 'nuc_AreaShape_Center_Y']]
        group = group.drop(columns=graph_features_to_remove)
        for img_type in ['cell_', 'nuc_']:
            for feat_type in ['RadialDistribution_FracAtD_E_', 'RadialDistribution_FracAtD_H_', 'RadialDistribution_MeanFrac_E_', 'RadialDistribution_MeanFrac_H_', 'RadialDistribution_RadialCV_E_', 'RadialDistribution_RadialCV_H_']:
                group[img_type + feat_type + 'max'] = group[[img_type + feat_type + x + 'of4' for x in '1234']].max(axis=1)
                group = group.drop(columns = [img_type + feat_type + x + 'of4' for x in '1234'])
        feature = group
        graphs.append(graph)
        features.append(feature)
    return graphs, features, ids

def process_sample_main(source, dest, fold_path):

    print('constructing database for 5-folds cross validation')
    folds = []
    for fold in '12345':
        img_dataset = dict()
        with open(os.path.join(fold_path, 'dsfold%s.txt' % fold)) as f:
            img_list = f.read().splitlines()
        for img in img_list:
            cancer_class, cancer_type, patient, p_id, magnitude, dataset_type = re.match(r'SOB_(?P<cancer_class>\w+)_(?P<cancer_type>\w+)-(?P<patient>[\w\d-]+)-\d+-(?P<id>\d+)\.png\|(?P<magnitude>\d+)\|\d+\|(?P<dataset>\w+)', img).groups()
            source_csv_path = os.path.join(source, cancer_class, cancer_type, patient)
            if source_csv_path not in img_dataset:
                img_dataset[source_csv_path] = dataset_type
        folds.append(img_dataset)
        print('done for fold %s' % fold)

    print('\nprocessing features')
    cancer_classes = [x for x in os.listdir(source) if os.path.isdir(os.path.join(source, x))]
    for c in cancer_classes:
        print('  processing class %s' % c)
        c_path = os.path.join(source, c)
        cancer_types = [x for x in os.listdir(c_path) if os.path.isdir(os.path.join(c_path, x))]
        for t in cancer_types:
            print('    processing type %s' % t)
            t_path = os.path.join(c_path, t)
            patients = [x for x in os.listdir(t_path) if os.path.isdir(os.path.join(t_path, x))]
            for p in patients:
                print('      processing patient %s' % p)
                source_csv_path = os.path.join(t_path, p)
                graphs, features, ids = process_sample(source_csv_path)
                for i in range(5):
                    fold = str(i + 1)
                    dataset_type = folds[i][source_csv_path]
                    dest_path = os.path.join(dest, 'fold%s' % fold, dataset_type, p)
                    os.makedirs(dest_path, exist_ok=True)
                    for index, graph in enumerate(graphs, 1):
                        graph.to_csv(os.path.join(dest_path, 'graph_%d.csv' % index))
                    for index, feature in enumerate(features, 1):
                        feature.to_csv(os.path.join(dest_path, 'feature_%d.csv' % index))

if __name__ == '__main__':
    process_sample_main(sys.argv[1], sys.argv[2], sys.argv[3])