import os
import subprocess
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import torch

import json
from collections import defaultdict

def dirichlet_split(dataset, alpha, num_clients):
        data_labels = np.array(dataset.targets)
        num_classes = len(np.unique(data_labels))
        client_indices = [[] for _ in range(num_clients)]

        for c in range(num_classes):
            class_indices = np.where(data_labels == c)[0]
            np.random.shuffle(class_indices)
            proportions = np.random.dirichlet([alpha] * num_clients)
            proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
            split_indices = np.split(class_indices, proportions)
            for i, indices in enumerate(split_indices):
                client_indices[i].extend(indices)

        client_datasets = [Subset(dataset, indices) for indices in client_indices]
        return client_datasets


def print_class_distribution(client_train_datasets, client_test_datasets, train_dataset, test_dataset):
    from collections import Counter

    # Getting the total number of classes
    total_classes = len(np.unique(train_dataset.targets))

    # Print class distribution for each client and check class set consistency
    print("Class Distribution and Consistency Check across Clients:")
    for i, (client_train, client_test) in enumerate(zip(client_train_datasets, client_test_datasets)):
        train_counts = Counter([train_dataset.targets[idx] for idx in client_train.indices])
        test_counts = Counter([test_dataset.targets[idx] for idx in client_test.indices])

        # Collect unique classes from counts
        train_classes = set(train_counts.keys())
        test_classes = set(test_counts.keys())

        # Check if training and testing datasets have the same set of classes
        same_classes = train_classes == test_classes

        print(f"\nClient {i+1} Data:")

        print(f"Training Data Distribution:")
        for j in range(total_classes):
            if train_counts[j] == 0:
                continue
            print(f"Class {j}: {train_counts[j]} samples")
        print(f"Testing Data Distribution:")
        for j in range(total_classes):
            if test_counts[j] == 0:
                continue
            print(f"Class {j}: {test_counts[j]} samples")

        # Print consistency check result
        print(f"Class Set Consistency between Training and Testing: {'Yes' if same_classes else 'No'}")


def get_cifar100_dataloader(args):
    # CIFAR-100 Dataset
    train_transform = transforms.Compose([ 
        transforms.RandomCrop(32, padding=4), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
    ])
    test_transform = transforms.Compose([ 
        transforms.ToTensor(), 
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
    ])

    train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=test_transform)

    # Partition dataset into 100 clients
    num_clients = args.num_clients
    if args.partitioner == "random":
        print("Random data partition is applied")
        client_train_datasets = random_split(train_dataset, [len(train_dataset) // num_clients] * num_clients)
        client_test_datasets = random_split(test_dataset, [len(test_dataset) // num_clients] * num_clients)
    elif args.partitioner == "iid":
        print("IID data partition is applied")
        data_per_client_train = len(train_dataset) // num_clients
        data_per_client_test = len(test_dataset) // num_clients
        client_train_datasets = [
            Subset(train_dataset, list(range(i * data_per_client_train, (i + 1) * data_per_client_train)))
            for i in range(num_clients)
        ]
        client_test_datasets = [
            Subset(test_dataset, list(range(i * data_per_client_test, (i + 1) * data_per_client_test)))
            for i in range(num_clients)
        ]
    elif args.partitioner == "noniid":
        print("Dirichlet data partition is applied")
        alpha = args.dirichlet_alpha

        client_train_datasets = dirichlet_split(train_dataset, alpha, num_clients)
        client_test_datasets = dirichlet_split(test_dataset, alpha, num_clients)
        
    else:
        print(f"[ERROR]: Specified data partition {args.partitioner} is not supported")
        exit()
    
    # Dataloader for each client (training and testing)
    client_train_loaders = [
        DataLoader(train_dataset, batch_size=args.bsz, shuffle=True) for train_dataset in client_train_datasets
    ]
    client_test_loaders = [
        DataLoader(test_dataset, batch_size=args.bsz, shuffle=False) for test_dataset in client_test_datasets
    ]

    return client_train_loaders, client_test_loaders

def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data

def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data
 

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
    