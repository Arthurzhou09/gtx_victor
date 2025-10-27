
from sagemaker.pytorch import PyTorch
from datetime import datetime

role =  'arn:aws:iam::425873948573:role/service-role/AmazonSageMaker-ExecutionRole-20220524T140113'
experiment_name = 'arthur-lightning-mcx-dropout-20-50'

model_name = "model_hikaru"
seed = 1024
train_subset = 5000  # <--- training subset of training data split. Test and val are split based on original train set size.(80/20/20)
activation = "relu"
optimizer = "Adam"
epochs = 60  # <--- epochs
nF = 6
learningRate = 1e-4 # <--- LR
batch = 16  # <--- batch size
xX = 101
yY = 101
decayRate = 0.4
depth_padding = 10
normalize = 0
fx_idx = "0 1 2 3 4 5"

scaleFL = 10e4
scaleOP0 = 10
scaleOP1 = 1
scaleDF = 1
scaleQF = 1
scaleRE = 1
nFilters3D = 128
kernelConv3D = "3 3 3"
strideConv3D = "1 1 1"
nFilters2D = 128
kernelConv2D = "3 3"
strideConv2D = "1 1"

data_path = "padded_DL_nImages1000_newOP_elecNoise.mat" # <--- Change this if you want to use a different dataset
bucket_name = "20250909-arthur" # <--- Change this to your own bucket

estimator = PyTorch(
    entry_point='run_train_py.py',
    source_dir='.',
    role=role,
    instance_count=1,                  
    instance_type='ml.g5.2xlarge',       
    framework_version='2.1',
    py_version='py310',
    dependencies=['requirements.txt'],
    hyperparameters={
        "seed": seed,
        "sagemaker": True,
        "train_subset": train_subset,
        "activation": "relu",
        "optimizer": "Adam",
        "epochs": epochs,
        "nF": nF,
        "learningRate": learningRate,
        "batch": batch,
        "xX": xX,
        "yY": yY,
        "decayRate": decayRate,
        "normalize": normalize,
        "depth_padding": depth_padding,
        "fx_idx": fx_idx,
        "scaleFL": scaleFL,
        "scaleOP0": scaleOP0,
        "scaleOP1": scaleOP1,
        "scaleDF": scaleDF,
        "scaleQF": scaleQF,
        "scaleRE": scaleRE,
        "nFilters3D": nFilters3D,
        "kernelConv3D": kernelConv3D,
        "strideConv3D": strideConv3D,
        "nFilters2D": nFilters2D,
        "kernelConv2D": kernelConv2D,
        "strideConv2D": strideConv2D,
        "data_path": data_path


    },
    output_path=f's3://{bucket_name}/output/'
)

job_name = f"{experiment_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
inputs = {
    'training': f's3://{bucket_name}/data/{data_path}',
}
estimator.fit(inputs=inputs, job_name=job_name)
