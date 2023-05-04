import torch
import random
import wandb
import torch.nn as nn
import torch.optim as optim

import lightning as L

from torchinfo import summary
from itertools import groupby
from eval_functions import get_metrics
from model.E2E_Score_Unfolding import get_fcn_model, get_rcnn_model, get_cnntrf_model

CONST_MODEL_IMPLEMENTATIONS = {
    "FCN": get_fcn_model,
    "CRNN": get_rcnn_model,
    "CNNT": get_cnntrf_model
}

class LighntingE2EModelUnfolding(L.LightningModule):
    def __init__(self, model, blank_idx, i2w, output_path) -> None:
        super(LighntingE2EModelUnfolding, self).__init__()
        self.model = model
        self.loss = nn.CTCLoss(blank=blank_idx)
        self.blank_idx = blank_idx
        self.i2w = i2w
        self.accum_ed = 0
        self.accum_len = 0
        
        self.dec_val_ex = []
        self.gt_val_ex = []
        self.img_val_ex = []
        self.ind_val_ker = []

        self.out_path = output_path

        self.save_hyperparameters(ignore=['model'])

    def forward(self, input):
        return self.model(input)
    
    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=1e-4)

    def training_step(self, train_batch, batch_idx):
         X_tr, Y_tr, L_tr, T_tr = train_batch
         predictions = self.forward(X_tr)
         loss = self.loss(predictions, Y_tr, L_tr, T_tr)
         self.log('loss', loss, on_epoch=True, batch_size=1, prog_bar=True)
         return loss

    def compute_prediction(self, batch):
        X, Y, _, _ = batch
        pred = self.forward(X)
        pred = pred.permute(1,0,2).contiguous()
        pred = pred[0]
        out_best = torch.argmax(pred,dim=1)
        out_best = [k for k, g in groupby(list(out_best))]
        decoded = []
        for c in out_best:
            if c.item() != self.blank_idx:
                decoded.append(c.item())
        
        decoded = [self.i2w[tok] for tok in decoded]
        gt = [self.i2w[int(tok.item())] for tok in Y[0]]

        return decoded, gt

    def validation_step(self, val_batch, batch_idx):
        dec, gt = self.compute_prediction(val_batch)
        
        dec = "".join(dec)
        dec = dec.replace("<t>", "\t")
        dec = dec.replace("<b>", "\n")
        dec = dec.replace("<s>", " ")

        gt = "".join(gt)
        gt = gt.replace("<t>", "\t")
        gt = gt.replace("<b>", "\n")
        gt = gt.replace("<s>", " ")

        self.dec_val_ex.append(dec)
        self.gt_val_ex.append(gt)

    def on_validation_epoch_end(self):        
        
        cer, ser, ler = get_metrics(self.dec_val_ex, self.gt_val_ex)

        self.log('val_CER', cer)
        self.log('val_SER', ser)
        self.log('val_LER', ler)

        return ser

    def test_step(self, test_batch, batch_idx):
        dec, gt = self.compute_prediction(test_batch)
        
        dec = "".join(dec)
        dec = dec.replace("<t>", "\t")
        dec = dec.replace("<b>", "\n")
        dec = dec.replace("<s>", " ")

        gt = "".join(gt)
        gt = gt.replace("<t>", "\t")
        gt = gt.replace("<b>", "\n")
        gt = gt.replace("<s>", " ")


        with open(f"{self.out_path}/hyp/{batch_idx}.krn", "w+") as krnfile:
            krnfile.write(dec)
        
        with open(f"{self.out_path}/gt/{batch_idx}.krn", "w+") as krnfile:
            krnfile.write(gt)

        self.dec_val_ex.append(dec)
        self.gt_val_ex.append(gt)
        self.img_val_ex.append((255.*test_batch[0].squeeze(0)))
    
    def on_test_epoch_end(self) -> None:
        cer, ser, ler = get_metrics(self.dec_val_ex, self.gt_val_ex)

        self.log('val_CER', cer)
        self.log('val_SER', ser)
        self.log('val_LER', ler)
        
        columns = ['Image', 'PRED', 'GT']
        data = []

        nsamples = len(self.dec_val_ex) if len(self.dec_val_ex) < 5 else 5
        random_indices = random.sample(range(len(self.dec_val_ex)), nsamples)

        for index in random_indices:
            data.append([wandb.Image(self.img_val_ex[index]), "".join(self.dec_val_ex[index]), "".join(self.gt_val_ex[index])])
        
        table = wandb.Table(columns= columns, data=data)
        
        self.logger.experiment.log(
            {'Test samples': table}
        )

        self.gt_val_ex = []
        self.dec_val_ex = []

        return ser

def get_model(maxwidth, maxheight, in_channels, out_size, blank_idx, i2w, model_name, output_path):
    model = CONST_MODEL_IMPLEMENTATIONS[model_name](maxwidth, maxheight, in_channels, out_size)
    lighningModel = LighntingE2EModelUnfolding(model=model, blank_idx=blank_idx, i2w=i2w, output_path=output_path)
    summary(lighningModel, input_size=([1, in_channels, maxheight, maxwidth]))
    return lighningModel, model