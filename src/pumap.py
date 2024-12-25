import numpy as np
from umap import UMAP
from warnings import warn, catch_warnings, filterwarnings
from numba import TypingError
import os
from umap.spectral import spectral_layout
from sklearn.utils import check_random_state
import codecs, pickle
from sklearn.neighbors import KDTree
from sklearn.datasets import load_digits

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    warn(
        """The umap.parametric_umap package requires PyTorch to be installed.
    You can install PyTorch at https://pytorch.org/get-started/locally/
    """
    )
    raise ImportError("umap.parametric_umap requires PyTorch") from None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ParametricUMAP(UMAP):
    def __init__(
            self,
            batch_size=None,
            dims=None,
            encoder=None,
            decoder=None,
            parametric_reconstruction=False,
            parametric_reconstruction_loss_fcn=None,
            parametric_reconstruction_loss_weight=1.0,
            autoencoder_loss=False,
            reconstruction_validation=None,
            global_correlation_loss_weight=0,
            pytorch_fit_kwargs={},
            **kwargs
    ):
        """
        Parametric UMAP subclassing UMAP-learn, based on PyTorch.
        There is also a non-parametric implementation contained within to compare
        with the base non-parametric implementation.

        Parameters
        ----------
        batch_size : int, optional
            size of batch used for batch training, by default None
        dims :  tuple, optional
            dimensionality of data, if not flat (e.g. (32x32x3 images for ConvNet), by default None
        encoder : nn.Module, optional
            The encoder PyTorch model, by default None
        decoder : nn.Module, optional
            The decoder PyTorch model, by default None
        parametric_reconstruction : bool, optional
            whether to use parametric reconstruction, by default False
        parametric_reconstruction_loss_fcn : function, optional
            loss function for parametric reconstruction, by default None
        parametric_reconstruction_loss_weight : float, optional
            weight for the reconstruction loss, by default 1.0
        autoencoder_loss : bool, optional
            whether to use autoencoder loss, by default False
        reconstruction_validation : tuple, optional
            validation data for reconstruction, by default None
        global_correlation_loss_weight : float, optional
            weight for global correlation loss, by default 0
        pytorch_fit_kwargs : dict, optional
            additional kwargs for PyTorch fit, by default {}
        """
        super(ParametricUMAP, self).__init__(**kwargs)

        self.batch_size = batch_size
        self.dims = dims
        self.encoder = encoder
        self.decoder = decoder
        self.parametric_reconstruction = parametric_reconstruction
        self.parametric_reconstruction_loss_fcn = parametric_reconstruction_loss_fcn
        self.parametric_reconstruction_loss_weight = parametric_reconstruction_loss_weight
        self.autoencoder_loss = autoencoder_loss
        self.reconstruction_validation = reconstruction_validation
        self.global_correlation_loss_weight = global_correlation_loss_weight
        self.pytorch_fit_kwargs = pytorch_fit_kwargs

        if self.encoder is None or self.decoder is None:
            raise ValueError("Encoder and decoder models must be provided for Parametric UMAP.")

        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()))

    def fit(self, X, y=None):
        if self.batch_size is None:
            self.batch_size = min(100, X.shape[0])

        dataset = TensorDataset(torch.Tensor(X))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.pytorch_fit_kwargs.get('epochs', 10)):
            total_loss = 0
            for batch in dataloader:
                X_batch = batch[0].to(device)
                self.optimizer.zero_grad()

                # Forward pass through the encoder
                encoded = self.encoder(X_batch)

                # Calculate UMAP loss
                umap_loss = self.calculate_umap_loss(encoded, X_batch)

                if self.autoencoder_loss:
                    # Forward pass through the decoder
                    decoded = self.decoder(encoded)
                    recon_loss = self.parametric_reconstruction_loss_fcn(decoded, X_batch)
                    loss = umap_loss + self.parametric_reconstruction_loss_weight * recon_loss
                else:
                    loss = umap_loss

                if self.global_correlation_loss_weight > 0:
                    global_corr_loss = self.calculate_global_correlation_loss(encoded, X_batch)
                    loss += self.global_correlation_loss_weight * global_corr_loss

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(
                f"Epoch {epoch + 1}/{self.pytorch_fit_kwargs.get('epochs', 10)}, Loss: {total_loss / len(dataloader)}")

        return self

    def calculate_umap_loss(self, encoded, X_batch):
        # Placeholder for UMAP-specific loss calculation
        # This should include UMAP's cross-entropy and other losses
        # Specific details depend on how UMAP calculates its loss
        return torch.tensor(0.0, requires_grad=True)

    def calculate_global_correlation_loss(self, encoded, X_batch):
        # Placeholder for global correlation loss calculation
        # The actual implementation depends on specific details
        return torch.tensor(0.0, requires_grad=True)

    def transform(self, X):
        self.encoder.eval()
        with torch.no_grad():
            X_tensor = torch.Tensor(X).to(device)
            encoded = self.encoder(X_tensor)
        return encoded.cpu().numpy()

    def inverse_transform(self, X):
        self.decoder.eval()
        with torch.no_grad():
            X_tensor = torch.Tensor(X).to(device)
            decoded = self.decoder(X_tensor)
        return decoded.cpu().numpy()

    def save(self, filepath):
        # Save the encoder and decoder models
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)

    def load(self, filepath):
        # Load the encoder and decoder models
        checkpoint = torch.load(filepath)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# Define the encoder and decoder models
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

    def forward(self, x):
        return self.network(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


# Instantiate the encoder and decoder
encoder = Encoder()
decoder = Decoder()

# Load your dataset
digits = load_digits()
X = digits.data
y = digits.target

# Define a reconstruction loss function
reconstruction_loss_fcn = nn.MSELoss()

# Instantiate the ParametricUMAP class
parametric_umap = ParametricUMAP(
    batch_size=64,
    encoder=encoder,
    decoder=decoder,
    parametric_reconstruction=True,
    parametric_reconstruction_loss_fcn=reconstruction_loss_fcn,
    pytorch_fit_kwargs={'epochs': 50}
)

# Fit the model
parametric_umap.fit(X)

# Transform data
X_transformed = parametric_umap.transform(X)
print("Transformed Data Shape:", X_transformed.shape)

# Optionally, inverse transform data
X_reconstructed = parametric_umap.inverse_transform(X_transformed)
print("Reconstructed Data Shape:", X_reconstructed.shape)

# Save the model
parametric_umap.save('parametric_umap_model.pth')

# Load the model
parametric_umap.load('parametric_umap_model.pth')
