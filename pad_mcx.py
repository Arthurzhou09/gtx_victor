import numpy as np
import mat73
from scipy.io import savemat, loadmat

def pad_with_sample_mean(arr, pad_width):
    sample_mean = arr.mean()
    padded_sample = np.pad(arr, pad_width=pad_width, mode='constant', constant_values=sample_mean)
    return np.stack(padded_sample)


if __name__ == "__main__":

    filename = 'DL_nImages1000_newOP_elecNoise.mat'
    mcx_data_path ="/mnt/c/Users/Arthur Zhou/Documents/Data/MCX/" + filename
    mat = mat73.loadmat(mcx_data_path)
    DF = mat['DF']
    QF = mat['QF']
    OP = mat['OP']
    RE = mat['RE']
    FL = mat['F']


    print(DF.shape, QF.shape, OP.shape, RE.shape, FL.shape)

    """  DF_pad = pad_with_sample_mean(DF, pad_width=((0,0), (0, 1), (0, 1)),)
    QF_pad = pad_with_sample_mean(QF, pad_width=((0,0), (0, 1), (0, 1)), )
    OP_pad = pad_with_sample_mean(OP, pad_width=((0,0),(0, 1), (0, 1), (0, 0)), )
    RE_pad = pad_with_sample_mean(RE, pad_width=((0,0), (0, 1), (0, 1), (0, 0)), )
    FL_pad = pad_with_sample_mean(FL, pad_width=((0,0), (0, 1), (0, 1), (0, 0)), )"""

    padded_data = {
        'DF': DF_pad,
        'QF': QF_pad,
        'OP': OP_pad,
        'RE': RE_pad,
        'F': FL_pad
    }
    print(DF_pad.shape, QF_pad.shape, OP_pad.shape, RE_pad.shape, FL_pad.shape)
    
    savemat('padded' + filename, padded_data)