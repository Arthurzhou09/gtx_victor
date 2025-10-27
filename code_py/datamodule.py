import lightning as pl
#from utils.preprocess.dt_data_preprocess import read_data #mine
from dataset import FluorescenceDataset
from torch.utils.data import DataLoader, random_split
from prepro import load_data
import numpy as np 
import os

class FluorescenceDataModule(pl.LightningDataModule):
    def __init__(self, params, scale_params, batch_size):
        """
        use the params in the dic for batch size
        """
        super().__init__()
        self.params = params
        self.scale_params = scale_params
        self.batch_size = batch_size

    def prepare_data(self):
        """ It is not recommended to assign state (self.x = y) in this method bc it is called on a single main process so states won't be avalable in other processes."""
        pass
    
    def setup(self, stage=None):
        if self.params['sagemaker']:
            filepath = os.path.join('/opt/ml/input/data/training', self.params['data_path'])
        else:
            filepath = self.params['data_path']
        
        self.data = load_data(filepath, self.scale_params, self.params['seed'], self.params['normalize'])
        #self.data = read_data(self.params['data_path'], self.scale_params)
        """ 
        if self.params.get('testpath', False):
            self.test_dataset = read_data(self.params['testpath'], self.scale_params)"""
        
        N = self.data['train']['fluorescence'].shape[0]
        if self.params['train_subset'] and 0 < self.params['train_subset'] < N:
            rng = np.random.RandomState(1024)
            idx = rng.choice(N, size=self.params['train_subset'], replace=False)
            for key, _ in self.data['train'].items():
                self.data['train'][key] = self.data['train'][key][idx]

                """ train_fluorescence       = self.data['train']['fluorescence'][idx]
                train_op                  = self.data['train']['optical_props'][idx]
                train_depth               = self.data['train']['depth'][idx]
                train_concentration_fluor = self.data['train']['concentration_fluor'][idx]
                train_reflectance         = self.data['train']['reflectance'][idx]"""

        self.train_dataset = FluorescenceDataset(self.data['train'])
        self.val_dataset = FluorescenceDataset(self.data['val'])
        self.test_dataset = FluorescenceDataset(self.data['test']) 
        print("datamodule is setup")


    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size,
                        shuffle=False, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size,
                        shuffle=False, num_workers=4, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size,
                        shuffle=False, num_workers=4, persistent_workers=True) 
    

class FluorescenceTestDataModule(pl.LightningDataModule):
    def __init__(self, params:dict, batch_size:int, scale_params:dict, test: bool =True, pad_depth: float =10.0):
        """
        use the params in the dic for batch size
        """
        super().__init__()
        self.batch_size = batch_size
        self.scaleparams = scale_params
        self.params = params
        self.test = test
        self.pad_depth = pad_depth

    def prepare_data(self):
        pass
    def setup(self, stage=None):
        self.test_dataset = load_data(self.params['testpath'], self.scaleparams, seed=self.params['seed'], normalize=self.params['normalize'], test =self.test)

        if self.pad_depth is not None: # pad background with depth padding of 10
            print(f"you are padding background (0) with a depth of {self.pad_depth}")
            depth = self.test_dataset['test']['depth']
            depth[depth==0] = self.pad_depth
            self.test_dataset['test']['depth'] = depth
        self.test_dataset = FluorescenceDataset(self.test_dataset['test'])

    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size,
                        shuffle=False, num_workers=4, persistent_workers=True)