import torch
import copy
import random
import torch.nn.functional as F

# Federated averaging
def federated_averaging(client_models):
    global_model = copy.deepcopy(client_models[0].state_dict())
    for key in global_model.keys():
        global_model[key] = torch.mean(
            torch.stack([client_model.state_dict()[key].float() for client_model in client_models]), dim=0
        )
    for client_model in client_models:
        client_model.load_state_dict(global_model)

# Federated averaging with sigmoid
def federated_averaging_sigmoid(client_models):
    global_model = copy.deepcopy(client_models[0].state_dict())
    for key in global_model.keys():
        if 'mask' in key:
            global_model[key] = torch.logit(torch.mean(
                torch.stack([torch.sigmoid(client_model.state_dict()[key].float()) for client_model in client_models]), dim=0
            ))
        else:
            global_model[key] = torch.mean(
                torch.stack([client_model.state_dict()[key].float() for client_model in client_models]), dim=0
            )
    for client_model in client_models:
        client_model.load_state_dict(global_model)

# Federated averaging with binary
def binary_sampling(mask):
    # print("\nbinary message sent\n")
    clipped_mask = torch.sigmoid(mask)
    binary_mask = torch.bernoulli(clipped_mask)
    return binary_mask

def federated_averaging_binary(client_models):
    global_model = copy.deepcopy(client_models[0].state_dict())
    for key in global_model.keys():
        if 'mask' in key:
            global_model[key] = torch.logit(torch.mean(
                torch.stack([binary_sampling(client_model.state_dict()[key].float()) for client_model in client_models]), dim=0
            ))
        else:
            global_model[key] = torch.mean(
                torch.stack([client_model.state_dict()[key].float() for client_model in client_models]), dim=0
            )
    for client_model in client_models:
        client_model.load_state_dict(global_model)

# Training loop
def train_hybrid_federated_split_standard(args, device, client_train_loaders, client_test_loaders, client_models,  server_model, client_optimizers, server_optimizer, logfile_name):
    log_f = open(logfile_name, 'a')
 
    for round in range(args.num_rounds):
        total_loss = 0

        client_indices = list(range(args.num_clients))
        num_clients_to_select = int(args.num_clients * args.participation_rate)
        selected_clients = random.sample(client_indices, num_clients_to_select)

        if args.verbose == "size":

            total_uploading_comm = 0  # Bytes
            total_SL_comm = 0  # Bytes


        # Each client trains locally and sends intermediate outputs to the server
        for client_id, client_model in enumerate(client_models):
            client_model.train()
            
            if client_id not in selected_clients:
                continue
            
            print(f"Client {client_id} starts training")

            if args.verbose == "size":

                total_forward_comm = 0  # Bytes
                total_backward_comm = 0  # Bytes

            for epoch in range(args.num_epochs):
            
                for batch_X, batch_y in client_train_loaders[client_id]:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)  # Move data to GPU
                    client_optimizer = client_optimizers[client_id]
                    client_optimizer.zero_grad()

                    # Step 1: Client forward pass
                    client_output = client_model(batch_X)
                    if args.verbose == "size":
                        client_output_clone = client_output.clone().detach().requires_grad_(True)
                        forward_bytes_per_element = client_output.element_size()
                        forward_memory_bytes = client_output.numel() * forward_bytes_per_element  # Total size in bytes
                        total_forward_comm += forward_memory_bytes  # Accumulate total forward overhead
                        print(f"Client {client_id} → Server | Forward Communication: {forward_memory_bytes / 1024:.2f} KB, bytes per element: {forward_bytes_per_element}")

                    # Step 2: Server forward pass and loss computation
                    server_model.train()
                    server_output = server_model(client_output)
                    loss = F.nll_loss(server_output, batch_y)
                    total_loss += loss.item()

                    if args.verbose == "size":
                        server_output_clone = server_model(client_output_clone)
                        loss_clone = F.nll_loss(server_output_clone, batch_y)

                    # Step 3: Backpropagation on server
                    server_optimizer.zero_grad()
                    loss.backward()
                    
                    
                    if args.verbose == "size":
                        grads = torch.autograd.grad(loss_clone, client_output_clone, retain_graph=True)[0]  # Get gradients w.r.t client_output
                        if grads is not None:
                            backward_bytes_per_element = grads.element_size()
                            backward_memory_bytes = grads.numel() * backward_bytes_per_element
                            total_backward_comm += backward_memory_bytes  # Accumulate total backward overhead
                            print(f"Server → Client {client_id} | Backward Communication: {backward_memory_bytes / 1024:.2f} KB, bytes per element: {forward_bytes_per_element}")
                        

                    server_optimizer.step()     

                    # Step 4: Backpropagation on client
                    client_optimizer.step()

                    


                print(f"Client {client_id}, Epoch {epoch} Loss: {loss:.4f}")

            if args.verbose == "size":
                # Convert total communication overhead to KB/MB
                total_comm_KB = (total_forward_comm + total_backward_comm) / 1024
                total_comm_MB = total_comm_KB / 1024

                print(f"🔹 Total Forward Communication: {total_forward_comm / (1024 * 1024):.2f} MB")
                print(f"🔹 Total Backward Communication: {total_backward_comm / (1024 * 1024):.2f} MB")
                print(f"🚀 Total Communication Overhead (Forward + Backward): {total_comm_MB:.2f} MB")

                total_SL_comm += total_comm_MB

                model_size_MB = get_model_size(client_model)
                total_uploading_comm += model_size_MB
                print(f"Client {client_id} Model Size: ({model_size_MB:.2f} MB)")

                model_size_MB = get_model_size(server_model)
                print(f"Server Model Size: ({model_size_MB:.2f} MB)")

        if args.verbose == "size":
            print(f"🚀 Total Uploading Communication Overhead : {total_uploading_comm:.2f} MB")
            print(f"🚀 Total SL Communication Overhead : {total_SL_comm:.2f} MB")

        # Federated averaging for client models
        if args.aggregation == "True":
            if args.training_method == "mask":
                if args.aggregation_method == "sigmoid":
                    federated_averaging_sigmoid(client_models)
                elif args.aggregation_method == "binary":
                    federated_averaging_binary(client_models)
                else:
                    federated_averaging(client_models)
            else:
                federated_averaging(client_models)

        print(f"Communication round {round + 1}, Total Loss: {total_loss / args.num_clients:.4f}")
        
        # Test client and server models together with client-specific test data
        print("\nTesting each client's model with the server model using client-specific test data:")
        client_accuracies = test_clients_with_test_data(device,client_models, server_model, client_test_loaders)

        # Display average accuracy across clients
        avg_accuracy = sum(client_accuracies) / len(client_accuracies)
        print(f"\nAverage Client Test Accuracy: {avg_accuracy:.2f}%\n")

        log_f.write('Round #{}\tTotal Loss:{}\tAvg_accu:{}\n'.format(round+1, total_loss / args.num_clients, avg_accuracy))
        log_f.flush()
        
    log_f.close()

# Testing loop
def test_clients_with_test_data(device, client_models, server_model, client_test_loaders):
    server_model.eval()  # Set server model to evaluation mode
    client_accuracies = []  # To store each client's accuracy

    for client_id, (client_model, test_loader) in enumerate(zip(client_models, client_test_loaders)):
        client_model.eval()  # Set the client model to evaluation mode
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                # Move data to GPU
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                # Step 1: Forward pass through client model
                client_output = client_model(batch_X)

                # Step 2: Forward pass through server model
                server_output = server_model(client_output)

                # Step 3: Compute predictions
                _, predicted = torch.max(server_output, 1)  # Get predicted class
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

        accuracy = 100 * correct / total if total > 0 else 0  # Avoid division by zero
        client_accuracies.append(accuracy)
        # print(f"Client {client_id} Test Accuracy: {accuracy:.2f}%")

    return client_accuracies

def get_model_size(model):
    total_size = sum(p.numel() * p.element_size() for p in model.parameters())  # Size in bytes
    total_buffer_size = sum(p.numel() * p.element_size() for p in model.buffers())  # Size in bytes
    total_num = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_num}")
    print(f"Total size in bytes: {total_size}")
    print(f"Total buffer size in bytes: {total_buffer_size}")
    total_size += total_buffer_size
    total_size_KB = total_size / 1024  # Convert to KB
    total_size_MB = total_size_KB / 1024  # Convert to MB
    return total_size_MB