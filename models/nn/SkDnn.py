import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from torch.optim.swa_utils import AveragedModel
from torch.optim.swa_utils import SWALR
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from train.torchutils import get_default_device
from models.base.PipelineWrapper import RTRegressor


class _DnnModel(nn.Module):
    def __init__(self, *, n_features, number_of_hidden_layers, dropout_between_layers, activation,
                 number_of_neurons_per_layer):
        super().__init__()
        self.dropout_p = dropout_between_layers
        layers = [nn.Linear(n_features, number_of_neurons_per_layer)]
        nn.init.zeros_(layers[0].bias)
        # Intermediate hidden layers
        for _ in range(1, number_of_hidden_layers):
            layer = nn.Linear(number_of_neurons_per_layer, number_of_neurons_per_layer)
            nn.init.zeros_(layer.bias)
            layers.append(layer)
        self.hidden_layers = nn.ModuleList(layers)
        # Output layer
        self.l_out = nn.Linear(number_of_neurons_per_layer, 1)
        nn.init.zeros_(self.l_out.bias)
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'swish':
            self.activation = F.silu

    def forward(self, x):
        for hidden_layer in self.hidden_layers:
            x = F.dropout(self.activation(hidden_layer(x)), self.dropout_p)
        return self.l_out(x)


class _SkDnn(BaseEstimator, RegressorMixin):
    def __init__(self, number_of_hidden_layers, dropout_between_layers, activation,
                 number_of_neurons_per_layer, lr, max_number_of_epochs, annealing_rounds, swa_epochs,
                 batch_size, device=get_default_device()):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)
        self.n_epochs = self.annealing_rounds * self.max_number_of_epochs

    def _init_hidden_model(self, n_features):
        self._model = _DnnModel(
            n_features=n_features, number_of_hidden_layers=self.number_of_hidden_layers,
            dropout_between_layers=self.dropout_between_layers, activation=self.activation,
            number_of_neurons_per_layer=self.number_of_neurons_per_layer,
        ).to(self.device)
        min_lr = 0.1 * self.lr
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self._optimizer, T_0=self.max_number_of_epochs, T_mult=1, eta_min=min_lr
        )
        self._swa_model = AveragedModel(self._model)
        self._swa_scheduler = SWALR(self._optimizer, swa_lr=min_lr)

    def fit(self, X, y):
        self._init_hidden_model(X.shape[1])
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).view(-1, 1))
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._model.train()
        train_iters = len(data_loader)
        for epoch in range(self.n_epochs):
            for i, (xb, yb) in enumerate(data_loader):
                self._batch_step(xb, yb)
                self._scheduler.step(epoch + i / train_iters)

        self._swa_model.train()
        for epoch in range(self.swa_epochs):
            for xb, yb in data_loader:
                self._batch_step(xb, yb)
            self._swa_model.update_parameters(self._model)
            self._swa_scheduler.step()

        return self

    def _batch_step(self, xb, yb):
        self._optimizer.zero_grad()
        pred = self._model(xb.to(self.device))
        loss = F.l1_loss(pred, target=yb.to(self.device))
        loss.backward()
        self._optimizer.step()

    def predict(self, X):
        self._model.eval()
        self._swa_model.eval()
        with torch.no_grad():
            return self._swa_model(torch.from_numpy(X).to(self.device)).cpu().numpy().flatten()

    def __getstate__(self):
        state = super().__getstate__().copy()
        if '_model' in state.keys():
            for key in ['_model', '_optimizer', '_scheduler', '_swa_model', '_swa_scheduler']:
                state[key] = state[key].state_dict()
        return state

    def __setstate__(self, state):
        self.__dict__ = state.copy()
        if '_model' in state.keys():
            self._init_hidden_model(state['_model']['l1.weight'].shape[1])
            for key in ['_model', '_optimizer', '_scheduler', '_swa_model', '_swa_scheduler']:
                torch_model = getattr(self, key)
                torch_model.load_state_dict(state.pop(key))
                setattr(self, key, torch_model)


class SkDnn(RTRegressor):
    def __init__(self, number_of_hidden_layers=2, dropout_between_layers=0, activation='gelu',
                 number_of_neurons_per_layer=512,
                 lr=3e-4, max_number_of_epochs=30, annealing_rounds=2, swa_epochs=20, batch_size=32, var_p=0.9,
                 device=get_default_device(),
                 use_col_indices='all',
                 binary_col_indices=None, transform_output=True):
        super().__init__(use_col_indices, binary_col_indices, var_p, transform_output)
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

    def _init_regressor(self):
        return _SkDnn(**self._rt_regressor_params())
