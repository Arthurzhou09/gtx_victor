
import sys
sys.path.insert(0,'/mnt/c/Users/Arthur Zhou/GitHub/gtx_victor')
print(sys.path)

from lightning.pytorch.tuner import Tuner
import argparse
import torch
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from pl_wrapper import Unet
from datamodule import FluorescenceDataModule
from dataset import FluorescenceDataset
import os

def get_arg_parser(): # taken from victor
    parser = argparse.ArgumentParser(description="Hyperparameter configuration for fluorescence imaging model.")

    parser.add_argument('--sagemaker', type=bool, default=False, help='SageMaker mode')
    parser.add_argument('--model_name', type=str, default='model_hikaru', help='Model name')
    parser.add_argument('--train_subset', type=int, default=8000, help='Train subset')
    parser.add_argument('--seed', type=int, default=1024, help='Seed')

    # General hyperparameters
    parser.add_argument('--activation', type=str, default='relu', help='Activation function')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer name')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--nF', type=int, default=6, help='Number of fluroescent spatial frequencies (fluorescent images)')
    parser.add_argument('--learningRate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--xX', type=int, default=101, help='Image width')
    parser.add_argument('--yY', type=int, default=101, help='Image height')
    parser.add_argument('--decayRate', type=float, default=0.3, help='Learning rate decay factor')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--normalize', type=int, default=0, help='Normalize data')
    parser.add_argument('--depth_padding', type=int, default=0, help='Depth padding')
    parser.add_argument('--fx_idx', type=int, nargs=6, default=[0, 1, 2, 3, 4, 5])
    # Scaling parameters
    parser.add_argument('--scaleFL', type=float, default=10e4, help='Scaling factor for fluorescence')
    parser.add_argument('--scaleOP0', type=float, default=10, help='Scaling for absorption coefficient (μa)')
    parser.add_argument('--scaleOP1', type=float, default=1, help='Scaling for scattering coefficient (μs)')
    parser.add_argument('--scaleDF', type=float, default=1, help='Scaling for depth')
    parser.add_argument('--scaleQF', type=float, default=1, help='Scaling for fluorophore concentration')
    parser.add_argument('--scaleRE', type=float, default=1, help='Scaling for reflectance (optional)')

    # 3D Conv parameters
    parser.add_argument('--nFilters3D', type=int, default=128)
    parser.add_argument('--kernelConv3D', type=int, nargs=3, default=[3,3,3])
    parser.add_argument('--strideConv3D', type=int, nargs=3, default=[1,1,1])

    # 2D Conv parameters
    parser.add_argument('--nFilters2D', type=int, default=128)
    parser.add_argument('--kernelConv2D', type=int, nargs=2, default=[3,3])
    parser.add_argument('--strideConv2D', type=int, nargs=2, default=[1,1])

    # Data path
    parser.add_argument('--data_path', type=str, default='../data/ts_2d_10000.mat')
    parser.add_argument('--model_dir', type=str, default='../code_tf/aws_ckpt/')
    return parser


if __name__ == "__main__":
    
    parser = get_arg_parser()
    args = parser.parse_args()
    params = vars(args) # default options, change in SageMaker script

    scale_params = {
        'fluorescence': params['scaleFL'],
        'mu_a': params['scaleOP0'],
        'mu_s': params['scaleOP1'],
        'depth': params['scaleDF'],
        'concentration_fluor': params['scaleQF'],
        'reflectance': params['scaleRE']
    }

    torch.set_float32_matmul_precision('medium')

    #load data
    data_mod = FluorescenceDataModule(params, scale_params, batch_size=params['batch'])
    data_mod.prepare_data()
    data_mod.setup()

    #pl model init
    model = Unet(params=params)

    #change this for saving chekcpoints and loss
    if params['sagemaker']:
        model_dir = f'/opt/ml/model'
        save_dir = os.path.join(model_dir, 'modelckpt')
    else:
        experiment_name ='testing_lightning_local'
        root_dir = "/mnt/c/Users/Arthur Zhou/Documents/DL/logger"
        save_dir =os.path.join(root_dir,experiment_name)
        os.makedirs(save_dir, exist_ok=True)


    """root_dir = "/mnt/c/Users/Arthur Zhou/Documents/ML/logger"
    save_dir =os.path.join(root_dir, f"{experiment_name}")"""

    
    #os.makedirs(save_dir, exist_ok=True)

    #experiment_name = " " + "_log"
    logger = CSVLogger(save_dir=save_dir, name='loss', )

    # training
    checkpoint_callback = ModelCheckpoint(dirpath=save_dir,
                                        save_last=True,)

    trainer = Trainer(callbacks=[checkpoint_callback],
                    accelerator="gpu",
                    log_every_n_steps=10,
                    logger=logger,
                    max_epochs=params['epochs']) #change this in params if you want to use params


    """ # tuner for the trainer
    tuner = Tuner(trainer)

    # Auto-scale batch size by growing it exponentially (default)
    tuner.scale_batch_size(model, mode="power",data_mod )"""


    trainer.fit(model, datamodule=data_mod)

