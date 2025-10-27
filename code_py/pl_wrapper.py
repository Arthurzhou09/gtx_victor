import lightning as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.unet import UnetModel

class Unet(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters()
        self.params = params
        self.model = UnetModel(params)
        self.mse_loss = torch.nn.MSELoss()
        self.mae_loss = torch.nn.L1Loss()


        self.test_outputs = []

    def forward(self, inOP, inFL):
        qf, df = self.model(inOP, inFL)
        return qf, df
    
    def training_step(self, batch, batch_idx):
        fluorescence, op, concentration_fluor, depth = batch
        print(f'Batch {batch_idx} of {len(batch)}') if batch_idx % 10 == 0 else None
        

        if batch_idx == 0:
            print(fluorescence.shape, op.shape, concentration_fluor.shape, depth.shape)
       

        pred_qf, pred_depth = self(op, fluorescence)
        loss_qf = self.mae_loss(pred_qf, concentration_fluor)
        loss_depth = self.mae_loss(pred_depth, depth)

        loss = loss_qf + loss_depth

        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('qf_loss', loss_qf, on_epoch=True, prog_bar=True, logger=True)
        self.log('depth_loss', loss_depth, on_epoch=True, prog_bar=True, logger=True)
        print(f"Total Train Loss: {loss.item()} | Pred QF Loss: {loss_qf.item()} | Pred Depth Loss: {loss_depth.item()}")

        return loss


    def validation_step(self, batch, batch_idx):
        fluorescence, op, concentration_fluor, depth = batch
        print(f'Batch {batch_idx} of {len(batch)}') if batch_idx % 10 == 0 else None
        

        if batch_idx == 0:
            print(fluorescence.shape, op.shape, concentration_fluor.shape, depth.shape)
        
        opt = self.optimizers() # needs to be one optimier
        lr = opt.param_groups[0]['lr']

        self.log('lr', lr, prog_bar=True, logger=True)

        pred_qf, pred_depth = self(op, fluorescence)
        loss_qf = self.mae_loss(pred_qf, concentration_fluor)
        loss_depth = self.mae_loss(pred_depth, depth)

        loss = loss_qf + loss_depth


        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_qf_loss', loss_qf, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_depth_loss', loss_depth, on_epoch=True, prog_bar=True, logger=True)
        print(f"LR: {lr} | Total Validation Loss: {loss.item()} | Pred QF Loss: {loss_qf.item()} | Pred Depth Loss: {loss_depth.item()}")
        return loss

    def test_step(self, batch, batch_idx):
        fluorescence, op, concentration_fluor, depth = batch
        print(f'Batch {batch_idx} of {len(batch)}') if batch_idx % 10 == 0 else None
        
        if batch_idx == 0:
            print(fluorescence.shape, op.shape, concentration_fluor.shape, depth.shape)
    
        pred_qf, pred_depth = self(op, fluorescence)

        #### note the padding
        print('pad before', torch.min(pred_depth), torch.max(pred_depth), torch.mean(pred_depth))

        #pred_depth[(-0.5 < pred_depth) & (pred_depth < 0.5)] = 10.0#pad predictions
        print('pad after', torch.min(pred_depth), torch.max(pred_depth), torch.mean(pred_depth))
        ####

        loss_qf = self.mae_loss(pred_qf, concentration_fluor)
        loss_depth = self.mae_loss(pred_depth, depth)

        loss = loss_qf + loss_depth

        print('pred_qf.shape:', pred_qf.shape, 'pred_depth.shape:', pred_depth.shape, 'concentration_fluor.shape:', concentration_fluor.shape, 'depth.shape:', depth.shape, 'fluorescence.shape:', fluorescence.shape, 'op.shape:', op.shape)
        self.test_outputs.append({
            "pred_qf": pred_qf.detach().cpu(),
            "pred_depth": pred_depth.detach().cpu(),
            "f_target": fluorescence.detach().cpu(),
            'depth_target': depth.detach().cpu(),
            'concentration': concentration_fluor.detach().cpu(),
        })

        self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_qf_loss', loss_qf, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_depth_loss', loss_depth, on_epoch=True, prog_bar=True, logger=True)
        print(f"Total Test Loss: {loss.item()} | Pred QF Loss: {loss_qf.item()} | Pred Depth Loss: {loss_depth.item()}")

        return loss
    
    def on_test_epoch_end(self):

        # Aggregate 
        pred_qf = torch.cat([out["pred_qf"] for out in self.test_outputs], dim=0).numpy()
        pred_depth = torch.cat([out["pred_depth"] for out in self.test_outputs], dim=0).numpy()
        y_f = torch.cat([out["f_target"] for out in self.test_outputs], dim=0).numpy()
        y_depth = torch.cat([out["depth_target"] for out in self.test_outputs], dim=0).numpy()
        y_concentration = torch.cat([out["concentration"] for out in self.test_outputs], dim=0).numpy()

        # Save for outside the model

        print(f"size of the pred_qf: {pred_qf.shape}, pred_depth: {pred_depth.shape}, y_f: {y_f.shape}, y_depth: {y_depth.shape}, y_concentration: {y_concentration.shape}")
        self.pred_qf = pred_qf
        self.pred_depth = pred_depth
        self.test_f = y_f #
        self.test_depth = y_depth
        self.test_concentration = y_concentration

        self.test_outputs.clear()



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.params["learningRate"])
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",  factor=self.params['decayRate'], patience=5),
            "monitor": "val_loss"
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}






