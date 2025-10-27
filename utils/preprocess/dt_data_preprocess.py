import scipy.io as sio
import numpy as np
import mat73

def scale_data(data_dict, params):
    scaled_data_dict = {}
    for key, items in data_dict.items():
        scaled_data_dict[key] = items * params[key]
    return scaled_data_dict

def normalization(fluorescence, optical_props):
    f = (fluorescence - np.mean(fluorescence, axis=(1,2,3), keepdims=True)) / \
                   (np.std(fluorescence, axis=(1,2,3), keepdims=True) + 1e-6)
    mu_a = optical_props[..., 0]
    mu_s = optical_props[..., 1]

    mu_a_mean = np.mean(mu_a, axis=(1,2), keepdims=True)
    mu_a_std = np.std(mu_a, axis=(1,2), keepdims=True)
    mu_a_norm = (mu_a - mu_a_mean) / (mu_a_std + 1e-6)

    mu_s_mean = np.mean(mu_s, axis=(1,2), keepdims=True)
    mu_s_std = np.std(mu_s, axis=(1,2), keepdims=True)
    mu_s_norm = (mu_s - mu_s_mean) / (mu_s_std + 1e-6)
    return f, mu_a_norm, mu_s_norm

def load_split_data(file_path):

    try:
        data = sio.loadmat(file_path)
    except:
        data = mat73.loadmat(file_path)


    data = {k: v for k, v in data.items() if not k.startswith('__')}

    splits = ['train', 'val', 'test']
    data_by_split = {split: {} for split in splits}

    for key, value in data.items():
        for split in splits:
            if key.startswith(split + '_'):
                field = key[len(split) + 1:]
                data_by_split[split][field] = value
    print("data by split", data_by_split.keys(), data_by_split)
    return data_by_split

def load_data(file_path, scale_params=None):
    data_by_split = load_split_data(file_path)

    result = {}

    for type in ['train', 'val', 'test']:
        fluorescence = data_by_split[type]['F']
        optical_props = data_by_split[type]['OP']
        depth = data_by_split[type]['DF']
        concentration_fluor = data_by_split[type]['QF']
        reflectance = data_by_split[type]['RE']

        f, mu_a_norm, mu_s_norm = normalization(fluorescence, optical_props)
        # scaled_data_dict = scale_data({
        #     'fluorescence': f,
        #     'reflectance': reflectance,
        #     'depth': depth, 
        #     'mu_a': mu_a_norm,
        #     'mu_s': mu_s_norm,
        #     'concentration_fluor': concentration_fluor
        #     }, scale_params)
        data_dict = {
            'fluorescence': f,
            'reflectance': reflectance,
            'depth': depth, 
            'mu_a': mu_a_norm,
            'mu_s': mu_s_norm,
            'concentration_fluor': concentration_fluor}
        
        result[type] = data_dict

    return result


def read_data(file_path, scale_params=None):
    try:
        data = sio.loadmat(file_path)
        data_key = [k for k in data.keys() if not k.startswith('___')] 

        assert len(data_key) == 1, "more than one struct in file, please reformat"

        fluorescence = data[data_key[0]].F
        optical_props = data[data_key[0]].OP
        depth = data[data_key[0]].DF
        concentration_fluor = data[data_key[0]].QF
        reflectance = data[data_key[0]].RE
    except:
        data = mat73.loadmat(file_path)
        fluorescence = data['F']
        optical_props = data['OP']
        depth = data['DF']
        concentration_fluor = data['QF']
        reflectance = data['RE']

    
    f, mu_a_norm, mu_s_norm = normalization(fluorescence, optical_props)

    data_dict = {
            'fluorescence': f,
            'reflectance': reflectance,
            'depth': depth, 
            'mu_a': mu_a_norm,
            'mu_s': mu_s_norm,
            'concentration_fluor': concentration_fluor}

    return data_dict