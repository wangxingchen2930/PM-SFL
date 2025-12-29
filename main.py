
import random
import torch
import torch.optim as optim
import numpy as np

import time
from datetime import datetime

from data.utils import get_cifar100_dataloader
from models.ResNet import ClientResNet18, ServerResNet18
from models.MaskedResNet import MaskedClientResNet18
from utils.args import get_args
from utils.standard_train import train_hybrid_federated_split_standard

# Set random seeds for reproducibility
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)


if __name__ == "__main__":

    start_time = time.time()

    args = get_args()

    set_random_seed(args.seed)
    
    # Check for GPU
    device = torch.device("cuda:" + str(args.ind_gpu) if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.dataset == "cifar100":

        # Dataloader
        client_train_loaders, client_test_loaders = get_cifar100_dataloader(args)

        if args.training_method == "mask":

            print("Training the mask")
            # Initialize client and server models
            client_models = [MaskedClientResNet18().to(device) for _ in range(args.num_clients)]  # Move models to GPU
            server_model = ServerResNet18(100).to(device)  # Move server model to GPU

            # Optimizers
            client_optimizers = [optim.Adam(client_model.parameters(), lr=args.lr) for client_model in client_models]
            server_optimizer = optim.Adam(server_model.parameters(), lr=args.lr)
        

        else:
            print("Training the weight")

            # Initialize client and server models
            client_models = [ClientResNet18().to(device) for _ in range(args.num_clients)]  # Move models to GPU
            server_model = ServerResNet18(100).to(device)  # Move server model to GPU

            # Optimizers
            client_optimizers = [optim.Adam(client_model.parameters(), lr=args.lr) for client_model in client_models]
            server_optimizer = optim.Adam(server_model.parameters(), lr=args.lr)
       

    else:
        print(f"[ERROR]: dataset {args.dataset} is not supported")
        exit()


    if args.aggregation != "True":
        args.aggregation_method = "None" 

    # Get current date and time
    now = datetime.now()
    # Format date and time
    formatted_time = now.strftime("%Y-%m-%d-%H-%M-%S")

    if args.partitioner == "noniid":
        logfile_name = './log/{}_{}Train_{}Aggr_{}_{}_alpha{}_N{}_PR{}_T{}_E{}_bsz{}.txt'.format(formatted_time, args.training_method, args.aggregation_method, args.dataset, args.partitioner, args.dirichlet_alpha, args.num_clients, args.participation_rate, args.num_rounds, args.num_epochs, args.bsz)
    else:
        logfile_name = './log/{}_{}Train_{}Aggr_{}_{}_N{}_PR{}_T{}_E{}_bsz{}.txt'.format(formatted_time, args.training_method, args.aggregation_method, args.dataset, args.partitioner, args.num_clients, args.participation_rate, args.num_rounds, args.num_epochs, args.bsz)

    # Train the model
    train_hybrid_federated_split_standard(args, device, client_train_loaders, client_test_loaders, client_models,  server_model, client_optimizers, server_optimizer, logfile_name)

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.4f} seconds")
