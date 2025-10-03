import tensorflow as tf
import scipy.io as sio
import mat73
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import keras
import tensorflow as tf
from keras.saving import register_keras_serializable
import sys
from preprocess import *
from joblib import Parallel, delayed
from tensorflow.python.client import device_lib



@register_keras_serializable(package="metrics_losses")
class TumorMAE(tf.keras.metrics.Metric):
    def __init__(self, depth_padding=10.0, name="tumor_mae", **kwargs):
        super().__init__(name=name, **kwargs)
        self.depth_padding = depth_padding
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = tf.not_equal(y_true, self.depth_padding)
        err  = tf.abs(y_true - y_pred)
        masked_err = tf.boolean_mask(err, mask)
        value = tf.cond(
            tf.size(masked_err) > 0,
            lambda: tf.reduce_mean(masked_err),
            lambda: tf.constant(0.0)
        )
        self.total.assign_add(value)
        self.count.assign_add(1.0)

    def result(self):
        return self.total / self.count
    

def test_load_model(model_path):
    model = keras.models.load_model(model_path, safe_mode=True)

    input_1 = tf.constant(np.ones((1, 101, 101, 2), dtype=np.float32))
    input_2 = tf.constant(np.ones((1, 101, 101, 6, 1), dtype=np.float32))

    output = model([input_1, input_2], training=False)

    print(output[0].shape)
    print(output[1].shape)


def draw_pred(gt_depth, pred_depth):
  fig, ax = plt.subplots(1, 3, figsize=(15, 5))
  v_min = min(np.min(gt_depth), np.min(pred_depth))
  v_max = max(np.max(gt_depth), np.max(pred_depth))
  im0 = ax[0].imshow(gt_depth, cmap='viridis', vmin=v_min, vmax=v_max)
  ax[0].set_title('Ground Truth')
  ax[1].imshow(pred_depth, cmap='viridis', vmin=v_min, vmax=v_max)
  ax[1].set_title('Prediction')
  im2 = ax[2].imshow(gt_depth - pred_depth, cmap='Purples')
  ax[2].set_title('Difference')
  cbar = fig.colorbar(im0, ax=ax[0:2], orientation='vertical')
  cbar.set_label('Depth')
  cbar_err = fig.colorbar(im2, ax=ax[2], orientation='vertical')
  cbar_err.set_label('Error')
  plt.show()


def load_phantom_data(path):
    def scale_data(data_dict, params):
        scaled_data_dict = {}
        for key, items in data_dict.items():
            if key != "optical_props" and key != "op":
                scaled_data_dict[key] = items * params[key]
        return scaled_data_dict

    data = mat73.loadmat(path)
    fluorescence = data['F']
    optical_props = data['OP']
    depth = data['DF']
    concentration_fluor = data['QF']
    reflectance = data['RE']

    output = {
        'fluorescence': fluorescence,
        'reflectance': reflectance,
        'depth': depth,
        'mu_a': optical_props[..., 0],
        'mu_s': optical_props[..., 1],
        'concentration_fluor': concentration_fluor
    }
    scale_params = {
    'fluorescence': 10e4,
    'mu_a': 10,
    'mu_s': 1,
    'depth': 1,
    'concentration_fluor': 1,
    'reflectance': 1,
    }
    

    return scale_data(output, scale_params)



from skimage.filters import threshold_otsu
def pred_phantom_1(model, phantom_data, idx, draw=False, depth_padding=10.0, thresh =True):
    fl = phantom_data['fluorescence'][[idx], ...]
    mua = phantom_data['mu_a'][[idx], ...]
    mus = phantom_data['mu_s'][[idx], ...]
    op = np.stack([mua, mus], axis=-1)
    conc_fluor = phantom_data['concentration_fluor'][[idx], ...]
    reflectance = phantom_data['reflectance'][[idx], ...]
    depth = phantom_data['depth'][[idx], ...]
    depth[depth==0] = depth_padding

    input1 = tf.convert_to_tensor(op, dtype=tf.float32)
    input2 = tf.convert_to_tensor(fl, dtype=tf.float32)

    outputs = model([input1, input2], training=False)

    pred_conc = outputs[0].numpy()
    pred_depth = outputs[1].numpy()

    if thresh:
      print("using otsu thresh")
      thresh = threshold_otsu(pred_depth[0])
      mask = pred_depth[0] < thresh
      masked_img = np.where(mask, pred_depth[0], 0)
    else:
      pred_depth_masked = np.where(pred_conc[0] < np.float32(0.1), 10.0, pred_depth[0])
      pred_depth_masked = np.where(pred_depth_masked >= np.float32(10.0), 0, pred_depth_masked)
      masked_img = pred_depth_masked

    gt = depth[0].copy()
    gt[gt==depth_padding]=0
    max_gt = np.max(gt[gt!=depth_padding])
    p = masked_img.copy()

    depth_arr = depth[0].flatten()
    pred_depth_arr = pred_depth[0].flatten()

    min_gt_value = np.min(depth_arr)
    min_gt_px = np.argmin(depth_arr)
    min_pred_value = np.min(pred_depth_arr)
    min_pred_px = np.argmin(pred_depth_arr)
    err_min = np.abs(min_gt_value-min_pred_value)
    print(f"Diff： {err_min}") if draw else None

    draw_pred(gt, masked_img) if draw else None

    return err_min, gt, masked_img, pred_conc[0], conc_fluor[0]


def parallel(model, phantom_data, thresh =False):
    #call
    length = len(phantom_data['fluorescence'])
    res = Parallel(n_jobs = 10, prefer = "threads", verbose = 10)( delayed(pred_phantom_1)(model, phantom_data, i, draw=False, depth_padding=10.0, thresh = thresh) 
                                for i in range(length))
    return res


"""
if __name__ == "__main__":
    

    print(tf.test.is_built_with_cuda())
    print(tf.config.list_physical_devices("GPU"))
    print("Python:", sys.executable)
    print("TF version:", tf.__version__)
    print("CUDA built with:", tf.sysconfig.get_build_info().get("cuda_version"))
    print("cuDNN built with:", tf.sysconfig.get_build_info().get("cudnn_version"))

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            print(gpus)
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)

    def get_available_devices():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos]

    print(get_available_devices())

    #test loading the model
    model_path_DT = "/mnt/c/Users/Arthur Zhou/GitHub/gtx_victor/code_tf/model_params/hikaru_dropout_2DnoiseDTTBR/model.keras"
    model_path_drive = "/content/drive/MyDrive/GTxPython/new_ckpt_unet/hikaru_2d_pad20/model.keras"
    test_load_model(model_path_DT)
    
    mcx_drop = '/mnt/c/Users/Arthur Zhou/GitHub/gtx_victor/code_tf/model_params/hikaru_dropout_pad10_noTBR/model.keras'
    mcx_no_drop = '/mnt/c/Users/Arthur Zhou/GitHub/gtx_victor/code_tf/model_params/hikaru_pad10_noTBR/model.keras'

    model = keras.models.load_model(mcx_no_drop, safe_mode=True)

    #load pahntoms
    drive_path = '/content/drive/MyDrive/GTxPython/data/phantom/phantom_data_corrected.mat'
    local_path = '/mnt/c/Users/Arthur Zhou/GitHub/gtx_victor/data/phantom/DL_nImages1.mat'   
    phantom_data = load_phantom_data(local_path)

    #call
    res = parallel(model, phantom_data, thresh =True)


"""