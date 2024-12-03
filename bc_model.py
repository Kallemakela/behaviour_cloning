import torch
import pytorch_lightning as pl


class BehavioralCloningModule(pl.LightningModule):
    def __init__(self, policy_network, lr=1e-3):
        super().__init__()
        # self.save_hyperparameters()
        self.policy_network = policy_network
        self.lr = lr
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        dist = self.policy_network.get_distribution(x)
        return dist.distribution.logits

    def training_step(self, batch, batch_idx=0):
        observations, actions = batch
        logits = self(observations)
        loss = self.loss_function(logits, actions.long())
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Use Adam optimizer for training
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class PretrainACModule(pl.LightningModule):
    def __init__(self, policy_network, lr=1e-3):
        super().__init__()
        # self.save_hyperparameters()
        self.policy_network = policy_network
        self.lr = lr
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        dist = self.policy_network.get_distribution(x)
        return dist.distribution.logits

    def training_step(self, batch, batch_idx=0):
        observations, actions = batch
        logits = self(observations)
        loss = self.loss_function(logits, actions.long())
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Use Adam optimizer for training
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
