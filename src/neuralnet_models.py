import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

import shap


def load_data(path: str):
    data = pd.read_csv(path, index_col=0)
    data.drop('player', axis=1, inplace=True)
    data.drop('rating', axis=1, inplace=True)
    data.reset_index(drop=True, inplace=True)
    data['agent_map'] = data.apply(lambda row: row['agent'] + ',' + row['map'], axis=1)
    data = pd.get_dummies(data, prefix=['agent_map'], columns=['agent_map'])
    data.drop(['map', 'agent'], axis=1, inplace=True)
    X = data.drop('won', axis=1).values.astype(float)
    y = data['won'].values
    return X, y


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(184, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        return x

    def forward_numpy(self, x):
        x = torch.tensor(x).float()
        x = self.forward(x)
        x = x.detach().numpy()
        return x


class Trainer:
    def __init__(self, model):
        self.model = model

    def fit(self, train_data, dev_data, batch_size=16, shuffle=True, epochs=5, lr=1e-3):
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=shuffle)

        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            self.train(train_loader, loss_fn, optimizer)
            self.test(dev_loader, loss_fn)
        print("Done!")

    def test(self, dev_loader, loss_fn):
        size = len(dev_loader.dataset)
        num_batches = len(dev_loader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dev_loader:
                X, y = X.float(), y.float()
                pred = model(X).squeeze(dim=1)

                test_loss += loss_fn(pred, y).item()
                correct += (torch.where(pred > 0.5, 1, 0) == y).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def train(self, train_loader, loss_fn, optimizer):
        size = len(train_loader.dataset)
        self.model.train()
        for batch, (X, y) in enumerate(train_loader):
            X, y = X.float(), y.float()

            # Compute prediction error
            pred = model(X).squeeze(dim=1)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


if __name__ == '__main__':
    # Load data
    X, y = load_data('data.csv')

    # Split into train/dev/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1)
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train, y_train, test_size=0.2)
    print(f'train: {X_train.shape}, {y_train.shape}')
    print(f'dev: {X_dev.shape}, {y_dev.shape}')
    print(f'test: {X_test.shape}, {y_test.shape}')

    # Convert NumPy/Pandas data to Tensor datasets
    train_data = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    dev_data = TensorDataset(torch.tensor(X_dev), torch.tensor(y_dev))
    test_data = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    model = NeuralNet()
    trainer = Trainer(model)
    trainer.fit(train_data, test_data, epochs=10)

    explainer = shap.Explainer(model.forward_numpy, X_train[:100])
    shap_values = explainer(X_test)

    plot1 = shap.summary_plot(shap_values, feature_names=[
        'Average Combat Score',
        'Kills',
        'Deaths',
        'Assists',
        'KAST',
        'Average Damage per Round',
        'Headshot %',
        '# First Kills',
        '# First Deaths',
        'Astra on Ascent',
        'Astra on Bind',
        'Astra on Breeze',
        'Astra on Fracture',
        'Astra on Haven',
        'Astra on Lotus',
        'Astra on Pearl',
        'Astra on Split',
        'Breach on ascent',
        'Breach on bind',
        'Breach on fracture',
        'Breach on haven',
        'Breach on lotus',
        'Breach on pearl',
        'Breach on split',
        'Breach on sunset',
        'Brimstone on Bind',
        'Brimstone on Fracture',
        'Brimstone on Icebox',
        'Brimstone on Lotus',
        'Chamber on Bind',
        'Chamber on Icebox',
        'Chamber on Lotus',
        'Chamber on Split',
        'Chamber on Sunset',
        'Cypher on Ascent',
        'Cypher on Bind',
        'Cypher on Breeze',
        'Cypher on Fracture',
        'Cypher on Haven',
        'Cypher on Icebox',
        'Cypher on Lotus',
        'Cypher on Pearl',
        'Cypher on Split',
        'Cypher on Sunset',
        'Deadlock on bind',
        'Deadlock on icebox',
        'Fade on ascent',
        'Fade on bind',
        'Fade on fracture',
        'Fade on haven',
        'Fade on icebox',
        'Fade on lotus',
        'Fade on pearl',
        'Fade on split',
        'Fade on sunset',
        'Gekko on ascent',
        'Gekko on bind',
        'Gekko on haven',
        'Gekko on icebox',
        'Gekko on lotus',
        'Gekko on pearl',
        'Gekko on split',
        'Gekko on sunset',
        'Harbor on ascent',
        'Harbor on bind',
        'Harbor on breeze',
        'Harbor on haven',
        'Harbor on icebox',
        'Harbor on lotus',
        'Harbor on pearl',
        'Harbor on sunset',
        'agent_map_jett,ascent',
        'agent_map_jett,bind',
        'agent_map_jett,breeze',
        'agent_map_jett,fracture',
        'agent_map_jett,haven',
        'agent_map_jett,icebox',
        'agent_map_jett,lotus',
        'agent_map_jett,pearl',
        'agent_map_jett,split',
        'agent_map_jett,sunset',
        'agent_map_kayo,ascent',
        'agent_map_kayo,bind',
        'agent_map_kayo,breeze',
        'agent_map_kayo,fracture',
        'agent_map_kayo,haven',
        'agent_map_kayo,icebox',
        'agent_map_kayo,lotus',
        'agent_map_kayo,pearl',
        'agent_map_kayo,split',
        'agent_map_kayo,sunset',
        'agent_map_killjoy,ascent',
        'agent_map_killjoy,fracture',
        'agent_map_killjoy,haven',
        'agent_map_killjoy,icebox',
        'agent_map_killjoy,lotus',
        'agent_map_killjoy,pearl',
        'agent_map_killjoy,split',
        'agent_map_neon,bind',
        'agent_map_neon,breeze',
        'agent_map_neon,fracture',
        'agent_map_neon,lotus',
        'agent_map_neon,pearl',
        'agent_map_neon,split',
        'agent_map_neon,sunset',
        'agent_map_omen,ascent',
        'agent_map_omen,bind',
        'agent_map_omen,breeze',
        'agent_map_omen,fracture',
        'agent_map_omen,haven',
        'agent_map_omen,icebox',
        'agent_map_omen,lotus',
        'agent_map_omen,split',
        'agent_map_omen,sunset',
        'agent_map_phoenix,ascent',
        'agent_map_phoenix,bind',
        'agent_map_phoenix,breeze',
        'agent_map_phoenix,icebox',
        'agent_map_phoenix,lotus',
        'agent_map_phoenix,pearl',
        'agent_map_phoenix,split',
        'agent_map_phoenix,sunset',
        'agent_map_raze,ascent',
        'agent_map_raze,bind',
        'agent_map_raze,fracture',
        'agent_map_raze,haven',
        'agent_map_raze,icebox',
        'agent_map_raze,lotus',
        'agent_map_raze,pearl',
        'agent_map_raze,split',
        'agent_map_raze,sunset',
        'agent_map_reyna,ascent',
        'agent_map_reyna,bind',
        'agent_map_reyna,icebox',
        'agent_map_reyna,lotus',
        'agent_map_reyna,pearl',
        'agent_map_sage,ascent',
        'agent_map_sage,bind',
        'agent_map_sage,haven',
        'agent_map_sage,icebox',
        'agent_map_sage,lotus',
        'agent_map_sage,pearl',
        'agent_map_sage,split',
        'agent_map_sage,sunset',
        'agent_map_skye,ascent',
        'agent_map_skye,bind',
        'agent_map_skye,breeze',
        'agent_map_skye,fracture',
        'agent_map_skye,haven',
        'agent_map_skye,icebox',
        'agent_map_skye,lotus',
        'agent_map_skye,pearl',
        'agent_map_skye,split',
        'agent_map_skye,sunset',
        'agent_map_sova,ascent',
        'agent_map_sova,bind',
        'agent_map_sova,breeze',
        'agent_map_sova,fracture',
        'agent_map_sova,haven',
        'agent_map_sova,icebox',
        'agent_map_sova,lotus',
        'agent_map_sova,pearl',
        'agent_map_sova,split',
        'agent_map_sova,sunset',
        'agent_map_viper,ascent',
        'agent_map_viper,bind',
        'agent_map_viper,breeze',
        'agent_map_viper,fracture',
        'agent_map_viper,haven',
        'agent_map_viper,icebox',
        'agent_map_viper,lotus',
        'agent_map_viper,pearl',
        'agent_map_viper,split',
        'agent_map_viper,sunset',
        'agent_map_yoru,ascent',
        'agent_map_yoru,bind',
        'agent_map_yoru,breeze',
        'agent_map_yoru,fracture',
        'agent_map_yoru,haven',
        'agent_map_yoru,icebox',
        'agent_map_yoru,lotus',
        'agent_map_yoru,pearl',
        'agent_map_yoru,split',
        'agent_map_yoru,sunset',
    ], max_display=8)
