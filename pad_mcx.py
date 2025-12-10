import numpy as np
import mat73
from scipy.io import savemat, loadmat
import os
def pad_with_sample_mean(arr, pad_width):
    sample_mean = arr.mean()
    padded_sample = np.pad(arr, pad_width, mode='constant', constant_values=0)
    return np.stack(padded_sample)


if __name__ == "__main__":

    filename = 'dl_t1_t18_mcx_pattern_ppix_elecnoise.mat'
    #'DL_nImages1000_newOP_elecNoise.mat'
    mcx_data_path ="/mnt/c/Users/Arthur Zhou/Documents/Data/MCX/" + filename

    file_drive = "C:\\Users\\Arthur Zhou\\UHN\\GTx - DalyShare (1)\\ImageData\\DL\\20250924_Arthur\\MCX\\2025_tumour1tumour18_noise_Arthur\\"
    file_drive = "C:\\Users\\Arthur Zhou\\UHN\\GTx - DalyShare (1)\\ImageData\\SL\\SFDIData\\_MandolinStudy\\20221024_MandolinTumorstudy\\20221205_Tumours_v3\\Dec7Phantoms\\"

    phantom_folders  = ["T12", "T16", "T5", "T21", "T11"]
    for i, folder in enumerate(phantom_folders):
        for type in ['Refl', 'Fl']:
            file = os.path.join(file_drive, folder, type, "SFDIrefl.mat")

            mat = loadmat(file) 

            image = mat['R_sample'] # for raw sfdi.mat. (x,y, 6)
            """DF = mat['DF']
            QF = mat['QF']
            OP = mat['OP']
            RE = mat['RE']
            FL = mat['F']"""


            """print(DF.shape, QF.shape, OP.shape, RE.shape, FL.shape)"""
            print(image.shape)

            y_pad = 520 - image.shape[0]
            x_pad = 696 - image.shape[1]


            image_pad = pad_with_sample_mean(image, pad_width=((x_pad//2, x_pad//2 + x_pad % 2), (y_pad//2, y_pad//2 + y_pad % 2), (0,0)),)

            """ DF_pad = pad_with_sample_mean(DF, pad_width=((0,0), (0, x_pad), (0, y_pad)),)
            QF_pad = pad_with_sample_mean(QF, pad_width=((0,0), (0, x_pad), (0, y_pad)), )
            OP_pad = pad_with_sample_mean(OP, pad_width=((0,0),(0, x_pad), (0, y_pad), (0, 0)), )
            RE_pad = pad_with_sample_mean(RE, pad_width=((0,0), (0, x_pad), (0, y_pad), (0, 0)), )
            FL_pad = pad_with_sample_mean(FL, pad_width=((0,0), (0, x_pad), (0, y_pad), (0, 0)), )"""

            """padded_data = {
                'DF': DF_pad,
                'QF': QF_pad,
                'OP': OP_pad,
                'RE': RE_pad,
                'F': FL_pad
            }"""
            """print(DF_pad.shape, QF_pad.shape, OP_pad.shape, RE_pad.shape, FL_pad.shape)"""

            padded = {'R_sample': image_pad}
            
            savemat(os.path.join(file_drive, folder, type, "padded_SFDIrefl.mat"), padded)

