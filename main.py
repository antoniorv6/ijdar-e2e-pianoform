import os
import gin
import fire
import wandb

from data import load_dataset, batch_preparation_ctc
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from ModelManager import get_model, LighntingE2EModelUnfolding
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


@gin.configurable
def main(train_path=None, val_path=None, test_path=None, encoding=None, model_name=None):
    outpath = f"out/GrandStaff_{encoding}/{model_name}"
    os.makedirs(outpath, exist_ok=True)
    os.makedirs(f"{outpath}/hyp", exist_ok=True)
    os.makedirs(f"{outpath}/gt", exist_ok=True)

    
    train_dataset, val_dataset, test_dataset = load_dataset(train_path, val_path, test_path, corpus_name=f"GrandStaff_{encoding}")

    _, i2w = train_dataset.get_dictionaries()

    train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=20, collate_fn=batch_preparation_ctc)
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=20, collate_fn=batch_preparation_ctc)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=20, collate_fn=batch_preparation_ctc)

    maxheight, maxwidth = train_dataset.get_max_hw()

    model, torchmodel = get_model(maxwidth=maxwidth, maxheight=maxheight, in_channels=1, blank_idx=len(i2w), out_size=train_dataset.vocab_size()+1, i2w=i2w, model_name=model_name, output_path=outpath)

    wandb_logger = WandbLogger(project='E2E_Pianoform', name=model_name)
    
    early_stopping = EarlyStopping(monitor='val_SER', min_delta=0.01, patience=5, mode="min", verbose=True)
    
    checkpointer = ModelCheckpoint(dirpath=f"weights/{encoding}/{model_name}", filename=f"{model_name}", 
                                   monitor="val_SER", mode='min',
                                   save_top_k=1, verbose=True)

    trainer = Trainer(max_epochs=10000, logger=wandb_logger, callbacks=[checkpointer, early_stopping])
    
    trainer.fit(model, train_dataloader, val_dataloader)

    model = LighntingE2EModelUnfolding.load_from_checkpoint(checkpointer.best_model_path, model=torchmodel)
    trainer.test(model, test_dataloader)
    wandb.finish()

def launch(config):
    gin.parse_config_file(config)
    main()

if __name__ == "__main__":
    fire.Fire(launch)