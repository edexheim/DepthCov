import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything

from depth_cov.core.NonstationaryGpModule import NonstationaryGpModule
from depth_cov.data.data_modules import DepthCovDataModule

# TODO: Params in a config file?
def train(batch_size, model_path=None):

  # Data module
  data_module = DepthCovDataModule(batch_size)

  # Train model
  if model_path is None:
    model = NonstationaryGpModule()
  else:
    model = NonstationaryGpModule.load_from_checkpoint(model_path)

  # Setup training
  checkpoint_callback = ModelCheckpoint(
    monitor="loss_val",
    dirpath="./models/",
    filename= "gp-{epoch:02d}-{loss_val:.4f}"
  )
  trainer = pl.Trainer( \
    callbacks=[checkpoint_callback],
    accumulate_grad_batches=1,
    gpus=1,
    limit_train_batches=0.01,
    max_epochs=1000,
    # overfit_batches=1
    # detect_anomaly=True,
    # log_every_n_steps=1
    limit_val_batches=0.01
  )

  trainer.fit(model, data_module)

if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    
    batch_size = 4

    model_path = None
    
    train(batch_size, model_path)