import torch
import pytorch_lightning as pl


class QNetworkCNN(pl.LightningModule):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.save_hyperparameters()
        self.lr = 1e-3
        self.gamma = 0.99
        self.conv1 = torch.nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = torch.nn.Linear(1024, 256)
        self.fc2 = torch.nn.Linear(256, num_actions)

    def forward(self, x):
        # print("input:", x.shape)
        x = torch.nn.functional.relu(self.conv1(x))
        # print("conv1:", x.shape)
        x = torch.nn.functional.relu(self.conv2(x))
        # print("conv2:", x.shape)
        x = torch.nn.functional.relu(self.conv3(x))
        # print("conv3:", x.shape)
        x = torch.nn.functional.relu(self.fc1(x.view(x.size(0), -1)))
        # print("fc1:", x.shape)
        x = self.fc2(x)
        # print("fc2:", x.shape)
        return x

    def training_step(self, batch, batch_idx=0):
        # Unpack batch
        states, actions, rewards, next_states, dones = batch

        # Compute Q(s, a) for taken actions
        q_values = self(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values using Bellman equation
        with torch.no_grad():
            next_q_values = self(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            targets = rewards.float() + self.gamma * max_next_q_values * (
                1 - dones.long()
            )

        # Compute loss
        loss = torch.nn.functional.mse_loss(q_values, targets)

        # Log loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx=0):
        # Unpack batch
        states, actions, rewards, next_states, dones = batch

        # Compute Q(s, a) for taken actions
        q_values = self(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values using Bellman equation
        with torch.no_grad():
            next_q_values = self(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            targets = rewards.float() + self.gamma * max_next_q_values * (
                1 - dones.long()
            )

        # Compute loss
        loss = torch.nn.functional.mse_loss(q_values, targets)

        # Log validation loss
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class PolicyNetworkCNN(pl.LightningModule):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.save_hyperparameters()
        self.lr = 1e-3
        self.conv1 = torch.nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = torch.nn.Linear(1024, 256)
        self.fc2 = torch.nn.Linear(256, num_actions)

    def forward(self, x):
        # Forward pass through the convolutional layers
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx=0):
        # Unpack batch
        states, actions, rewards, next_states, dones = batch

        # Forward pass to get action probabilities
        action_probs = torch.nn.functional.softmax(self(states), dim=1)
        chosen_action_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        epsilon = 1e-8
        chosen_action_probs = chosen_action_probs.clamp(min=epsilon)
        # Compute loss - using negative log likelihood with rewards as weights
        loss = -torch.mean(torch.log(chosen_action_probs) * rewards.float())

        # Log loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx=0):
        # Unpack batch
        states, actions, rewards, next_states, dones = batch

        # Forward pass to get action probabilities
        action_probs = torch.nn.functional.softmax(self(states), dim=1)
        chosen_action_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        epsilon = 1e-8
        chosen_action_probs = chosen_action_probs.clamp(min=epsilon)

        # Compute loss - using negative log likelihood with rewards as weights
        loss = -torch.mean(torch.log(chosen_action_probs) * rewards.float())

        # Log validation loss
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
