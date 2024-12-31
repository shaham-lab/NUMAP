import numpy as np
from sklearn.datasets import fetch_openml
import clip
from PIL import Image
from sklearn.preprocessing import StandardScaler
import torch, torchvision
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from ucimlrepo import fetch_ucirepo
from sklearn.datasets import load_iris, load_breast_cancer, load_digits
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import pyreadr
import rdata
from scipy.sparse import csr_matrix


def add_noise(X, noise=0.1, random_state=None):
    np.random.seed(random_state)
    return X + np.random.normal(0, noise, X.shape)


def get_circle(n_samples=1000, noise=0.1, random_state=None):
    t = np.random.rand(n_samples) * 2 * np.pi
    X = np.array([np.cos(t), np.sin(t)]).T
    X = add_noise(X, noise, random_state)
    return torch.Tensor(X), t, 'circle'


def get_2circles2D(n_samples=2000, noise=0.1, random_state=None):
    t1 = np.random.rand(n_samples // 2) * 2 * np.pi
    t2 = np.random.rand(n_samples // 2) * 2 * np.pi
    x1 = np.array([np.cos(t1) - 1.5, np.sin(t1)]).T
    x2 = np.array([np.cos(t2) + 1.5, np.sin(t2)]).T
    X = np.concatenate([x1, x2])
    X = add_noise(X, noise, random_state)
    y = np.concatenate([np.zeros(n_samples // 2), np.ones(n_samples // 2)]).astype(int)
    return torch.Tensor(X), y, '2circles2D'


def get_2circles(n_samples=5000, noise=0.1, random_state=None):
    t1 = np.random.rand(n_samples // 2) * 2 * np.pi
    t2 = np.random.rand(n_samples // 2) * 2 * np.pi
    x1 = np.array([np.cos(t1), np.sin(t1), np.zeros(n_samples // 2)]).T
    x2 = np.array([np.zeros(n_samples // 2), 1 + np.cos(t2), np.sin(t2)]).T
    X = np.concatenate([x1, x2])
    X = add_noise(X, noise, random_state)
    y = np.concatenate([np.zeros(n_samples // 2), np.ones(n_samples // 2)]).astype(int)
    return torch.Tensor(X), y, '2circles'


def get_curvedLine(n_samples=1000, noise=0.1, random_state=None):
    """Generate a closed curve in 3D."""
    t = np.random.rand(n_samples) * 2 * np.pi
    X = np.array([np.cos(3 * t), np.sin(t), np.sin(2 * t)]).T
    X = add_noise(X, noise, random_state)
    return torch.Tensor(X), t, 'curvedLine'


def get_sphereInSphere(n_samples=2000, noise=0, random_state=None):
    """Generate a sphere in a sphere in 3D."""
    theta1 = np.random.rand(n_samples // 2) * 2 * np.pi
    theta2 = np.random.rand(n_samples // 2) * 2 * np.pi
    phi1 = np.random.rand(n_samples // 2) * np.pi
    phi2 = np.random.rand(n_samples // 2) * np.pi
    x1 = 0.5 * np.array([np.cos(theta1) * np.cos(phi1), np.cos(theta1) * np.sin(phi1), np.sin(theta1)]).T
    x2 = np.array([np.cos(theta2) * np.cos(phi2), np.cos(theta2) * np.sin(phi2), np.sin(theta2)]).T
    X = np.concatenate([x1, x2])
    X = add_noise(X, noise, random_state)
    y = np.concatenate([np.zeros(n_samples // 2), np.ones(n_samples // 2)]).astype(int)
    return torch.Tensor(X), y, 'sphereInSphere'


def get_planeSphere(n_samples=2000, noise=0, random_state=None):
    """Generate a plane and a sphere in 3D."""
    theta1 = np.random.rand(n_samples // 2) * 2 * np.pi
    phi1 = np.random.rand(n_samples // 2) * np.pi
    sphere = np.array([np.cos(theta1) * np.cos(phi1), np.cos(theta1) * np.sin(phi1), np.sin(theta1)]).T
    plane = np.random.rand(n_samples // 2, 2) * 2 - 1
    plane = np.concatenate([plane, -np.ones((n_samples // 2, 1))], axis=1)
    X = np.concatenate([sphere, plane])
    X = add_noise(X, noise, random_state)
    y = np.concatenate([np.zeros(n_samples // 2), np.ones(n_samples // 2)]).astype(int)
    return torch.Tensor(X), y, 'planeSphere'


def get_tangentSpheres(n_samples=2000, noise=0, random_state=None):
    """Generate two tangent spheres in 3D."""
    theta1 = np.random.rand(n_samples // 2) * 2 * np.pi
    theta2 = np.random.rand(n_samples // 2) * 2 * np.pi
    phi1 = np.random.rand(n_samples // 2) * np.pi
    phi2 = np.random.rand(n_samples // 2) * np.pi
    x1 = np.array([np.cos(theta1) * np.cos(phi1), np.cos(theta1) * np.sin(phi1), np.sin(theta1)]).T
    x2 = np.array([np.cos(theta2) * np.cos(phi2) + 2, np.cos(theta2) * np.sin(phi2), np.sin(theta2)]).T
    X = np.concatenate([x1, x2])
    X = add_noise(X, noise, random_state)
    y = np.concatenate([np.zeros(n_samples // 2), np.ones(n_samples // 2)]).astype(int)
    return torch.Tensor(X), y, 'tangentSpheres'


def get_cyls(n_samples=3000, noise=0, random_state=None):
    """Generate 3 cylinders in 3D."""
    r1, r2, r3 = 1, 0.5, 0.25
    theta1 = np.random.rand(n_samples // 3) * 2 * np.pi
    theta2 = np.random.rand(n_samples // 3) * 2 * np.pi
    theta3 = np.random.rand(n_samples // 3) * 2 * np.pi
    z1 = np.random.rand(n_samples // 3) * 2 - 1
    z2 = np.random.rand(n_samples // 3) * 2 - 1
    z3 = np.random.rand(n_samples // 3) * 2 - 1
    x1 = np.array([r1 * np.cos(theta1), r1 * np.sin(theta1), z1]).T
    x2 = np.array([r2 * np.cos(theta2), r2 * np.sin(theta2), z2]).T
    x3 = np.array([r3 * np.cos(theta3), r3 * np.sin(theta3), z3]).T
    X = np.concatenate([x1, x2, x3])
    X = add_noise(X, noise, random_state)
    y = np.concatenate([np.zeros(n_samples // 3), np.ones(n_samples // 3), 2 * np.ones(n_samples // 3)]).astype(int)
    return torch.Tensor(X), y, '3cyls'


def get_mnist():
    # X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    # X = StandardScaler().fit_transform(X)
    # return torch.Tensor(X), y.astype(int).values, 'mnist'
    tensor_transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(
        root="../data", train=True, download=True, transform=tensor_transform
    )
    test_set = datasets.MNIST(
        root="../data", train=False, download=True, transform=tensor_transform
    )

    x_train, y_train = zip(*train_set)
    x_train, y_train = torch.cat(x_train), torch.Tensor(y_train)
    x_test, y_test = zip(*test_set)
    x_test, y_test = torch.cat(x_test), torch.Tensor(y_test)

    x_train = x_train.view(x_train.size(0), -1)
    x_test = x_test.view(x_test.size(0), -1)

    return (x_train, y_train, x_test, y_test), 'mnist'


def get_fashionMnist():
    # X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True)
    # X = StandardScaler().fit_transform(X)
    # return torch.Tensor(X), y.astype(int).values, 'fashionMnist'
    tensor_transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.FashionMNIST(
        root="../data", train=True, download=True, transform=tensor_transform
    )
    test_set = datasets.FashionMNIST(
        root="../data", train=False, download=True, transform=tensor_transform
    )

    x_train, y_train = zip(*train_set)
    x_train, y_train = torch.cat(x_train), torch.Tensor(y_train)
    x_test, y_test = zip(*test_set)
    x_test, y_test = torch.cat(x_test), torch.Tensor(y_test)

    x_train = x_train.view(x_train.size(0), -1)
    x_test = x_test.view(x_test.size(0), -1)

    return (x_train, y_train, x_test, y_test), 'fashionMnist'


def get_cifar10(batch_size=256, classes=None):
    # Fetch CIFAR-10 dataset
    X, y = fetch_openml('CIFAR_10', version=1, return_X_y=True, as_frame=False)
    y = y.astype(int)

    # Normalize pixel values to the range [0, 1]
    X = X / 255.0

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Transform images to the format expected by CLIP
    def preprocess_image(image):
        image = image.reshape(3, 32, 32).transpose(1, 2, 0).astype(np.float32)
        image = Image.fromarray((image * 255).astype(np.uint8))
        return preprocess(image).unsqueeze(0).to(device)

    # Process images in batches
    n = X.shape[0]
    batches = (n + batch_size - 1) // batch_size  # Compute the number of batches
    image_features_list = []

    for i in range(batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, n)
        images_batch = torch.cat([preprocess_image(img) for img in X[start:end]])

        with torch.no_grad():
            batch_features = model.encode_image(images_batch)
            image_features_list.append(batch_features.cpu().numpy())

    # Concatenate all batch features
    image_features = np.concatenate(image_features_list, axis=0)

    if classes is not None:
        inds = np.where(y == classes[0])
        for one_class in classes[1:]:
            more_inds = np.where(y == one_class)
            print('inds', inds)
            print('more_inds', more_inds)
            inds = np.hstack((inds, more_inds))
        inds = inds.reshape(-1)
        y = y[inds]
        image_features = image_features[inds]

    # Normalize the image features
    # scaler = StandardScaler()
    # image_features = scaler.fit_transform(image_features)

    return torch.Tensor(image_features), torch.Tensor(y), 'cifar10'


def get_appliances():
    dataset = fetch_ucirepo(id=374)     # Appliances Energy Prediction, delete first column!!!
    dataset_name = "appliances"
    X, y = dataset.data.features.values, dataset.data.targets.values.reshape(-1)
    X = X[:, 1:].astype(np.float32)
    X = X[:20000]

    return torch.Tensor(X), y, 'appliances'


def get_parkinsons():
    dataset = fetch_ucirepo(id=189)  # Parkinsons Telemonitoring
    dataset_name = "parkinsons"
    X, y = dataset.data.features.values, dataset.data.targets.values
    y = y[:, 1].reshape(-1)

    # y = np.zeros(X.shape[0])
    # y[:1000] = 1

    return torch.Tensor(X), y, 'parkinsons'


def get_wine():
    dataset = fetch_ucirepo(id=109)  # Parkinsons Telemonitoring
    dataset_name = "wine"
    X, y = dataset.data.features.values, dataset.data.targets.values.reshape(-1)

    return torch.Tensor(X), y, 'wine'


def get_iris():
    iris = load_iris()

    # Convert the features (X) to a PyTorch tensor
    X = torch.tensor(iris['data'], dtype=torch.float32)

    # Convert the labels (y) to a NumPy array
    y = np.array(iris['target'])

    return X, y, 'iris'


def get_breast():
    breast = load_breast_cancer()

    # Convert the features (X) to a PyTorch tensor
    X = torch.tensor(breast['data'], dtype=torch.float32)

    # Convert the labels (y) to a NumPy array
    y = np.array(breast['target'])

    return X, y, 'breast'


def get_digits():
    digits = load_digits()

    # Convert the features (X) to a PyTorch tensor
    X = torch.tensor(digits['data'], dtype=torch.float32)

    # Convert the labels (y) to a NumPy array
    y = np.array(digits['target'])

    return X, y, 'digits'


def get_pendigits():
    dataset = fetch_ucirepo(id=81)
    dataset_name = "pendigits"
    X, y = dataset.data.features.values, dataset.data.targets.values.reshape(-1)

    return torch.Tensor(X), y, 'pendigits'


def get_banknote():
    dataset = fetch_ucirepo(id=267)
    dataset_name = "banknote"
    X, y = dataset.data.features.values, dataset.data.targets.values.reshape(-1)

    return torch.Tensor(X), y, 'banknote'


def get_glass():
    dataset = fetch_ucirepo(id=42)
    dataset_name = "glass"
    X, y = dataset.data.features.values, dataset.data.targets.values.reshape(-1)

    return torch.Tensor(X), y, 'glass'


def get_ecoli():
    dataset = fetch_ucirepo(id=39)
    dataset_name = "ecoli"
    X, y = dataset.data.features.values, dataset.data.targets.values.reshape(-1)

    le = LabelEncoder()
    y = le.fit_transform(y)

    return torch.Tensor(X), y, 'ecoli'


def get_haberman():
    dataset = fetch_ucirepo(id=43)
    dataset_name = "haberman"
    X, y = dataset.data.features.values, dataset.data.targets.values.reshape(-1)

    return torch.Tensor(X), y, 'haberman'


def get_heart():
    dataset = fetch_ucirepo(id=145)
    dataset_name = "heart"
    X, y = dataset.data.features.values, dataset.data.targets.values.reshape(-1)

    return torch.Tensor(X), y, 'heart'


def get_han(subset_size=10000, use_pca=True):
    # Load the RData file
    data = rdata.read_rda('rdata/han/xp.RData')
    data = data['xp']

    # Create a sparse matrix from the components
    sparse_matrix = csr_matrix((data.x, data.i, data.p), shape=(data.Dim[1], data.Dim[0]))
    sparse_matrix = sparse_matrix.transpose()

    n_rows = sparse_matrix.shape[0]
    random_rows = np.random.choice(n_rows, size=subset_size, replace=False)
    subset_matrix = sparse_matrix[random_rows, :]

    X = subset_matrix.toarray()

    # pca
    pca = PCA(n_components=100)
    X = pca.fit_transform(X)

    tissues = rdata.read_rda('rdata/han/tissues.RData')
    tissues = tissues['tissues']

    le = LabelEncoder()
    y = le.fit_transform(tissues)
    y = y[random_rows]

    return torch.from_numpy(X).float(), y, 'han'


def get_wong(subset_size=10000, use_pca=True):
    # Load the RData file
    data = rdata.read_rda('rdata/wong/xp.RData')
    data = data['xp'].values

    n_rows = data.shape[0]
    random_rows = np.random.choice(n_rows, size=subset_size, replace=False)
    X = data[random_rows, :]

    populations = rdata.read_rda('rdata/wong/populations.RData')
    y = populations['populations'].astype(int)
    y = y[random_rows]

    return torch.from_numpy(X).float(), y, 'wong'


def get_samusik(subset_size=10000, use_pca=True):
    # Load the RData file
    data = rdata.read_rda('rdata/samusik/xp.RData')
    # for key in data.keys():
    #     print(f"{key}: {data[key]}")
    data = data['xp'].values

    n_rows = data.shape[0]
    random_rows = np.random.choice(n_rows, size=subset_size, replace=False)
    X = data[random_rows, :]

    populations = rdata.read_rda('rdata/samusik/populations.RData')
    y = populations['populations'].astype(int)
    y = y[random_rows]

    return torch.from_numpy(X).float(), y, 'samusik'


def get_kmnist():
    # Define a transform to convert images to tensors and flatten them
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten the 28x28 image to a 784-dimensional vector
    ])

    # Load the dataset
    train_dataset = torchvision.datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transform)

    # Combine train and test datasets
    combined_dataset = ConcatDataset([train_dataset, test_dataset])

    # Create a DataLoader
    dataloader = DataLoader(combined_dataset, batch_size=1000, shuffle=False)

    # Initialize a list to store all samples
    all_samples = []
    all_labels = []

    # Iterate through the DataLoader
    for batch, labels in dataloader:
        all_samples.append(batch)
        all_labels.append(labels)

    # Stack all samples into a single tensor
    all_samples_tensor = torch.cat(all_samples, dim=0)
    all_labels_numpy = np.concatenate([label.numpy() for label in all_labels])

    return all_samples_tensor, all_labels_numpy, 'kmnist'
