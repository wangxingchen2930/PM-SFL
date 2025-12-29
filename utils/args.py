import argparse

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--ind_gpu', type=int, default=0)

    parser.add_argument('--num_clients', type=int, default=100)
    parser.add_argument('--participation_rate', type=float, default=0.1)

    parser.add_argument('--dataset', type=str, default="cifar100")
    parser.add_argument('--partitioner', type=str, default="noniid") # choices include "random", "iid", and "noniid"
    parser.add_argument('--dirichlet_alpha', type=float, default=0.3) # Smaller alpha values lead to more imbalanced splits.
    
    parser.add_argument('--personalization', type=bool, default=False)

    parser.add_argument('--bsz', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--num_epochs', type=int, default=5) # 1 for acc upper bounds
    parser.add_argument('--num_rounds', type=int, default=1000)

    parser.add_argument('--training_method', type=str, default="standard")
    parser.add_argument('--self_distillation', type=bool, default=False)
    parser.add_argument('--distillation_alpha', type=float, default=1)
    parser.add_argument('--sd_method', type=str, default="KL")
    parser.add_argument('--compensation', type=str, default="True")

    parser.add_argument('--aggregation', type=str, default="True")
    parser.add_argument('--aggregation_method', type=str, default="standard")

    parser.add_argument('--r_mask', type=int, default=1000)
    parser.add_argument('--num_class_per_client', type=int, default=10)

    parser.add_argument('--verbose', type=str, default="None")
    
    parser.add_argument('--malicious_fraction', type=float, default=0.1)
    parser.add_argument('--flip_prob', type=float, default=1.0)
    parser.add_argument('--num_classes', type=int, default=100)

    args = parser.parse_args()

    return args