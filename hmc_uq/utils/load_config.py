import argparse
import yaml

def load_config(config_file = 'configs/models/baseline.yaml', config_name='CYP'):
    """Loads a specific configuration from a YAML file."""
    with open(config_file, "r") as f:
        all_configs = yaml.safe_load(f)
    
    if config_name not in all_configs:
        raise ValueError(f"Config '{config_name}' not found in {config_file}. Available configs: {list(all_configs.keys())}")
    
    return all_configs[config_name]

def get_args(config_file = '/configs/models/baseline.yaml', config_name = 'CYP'):
    """Parses command-line arguments, with defaults from a YAML file."""
    parser = argparse.ArgumentParser(description="Model Configs")

    # Define model arguments
    #defaults overwrite confis in yaml files
    parser.add_argument("--TargetID", type=str)
    parser.add_argument("--tr_fold", type=list, default=[2, 3, 4])
    parser.add_argument("--va_fold", type=int, default=1)
    parser.add_argument("--te_fold", type=int, default=0)
    parser.add_argument("--model_loss", type=str, default='BCELoss')
    #Model HPs
    parser.add_argument("--nr_layers", type=int)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--hidden_sizes", type=int)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--dropout", type=float)
    #BBB HPs
    parser.add_argument("--prior_mu", type=float, default=0)
    parser.add_argument("--prior_rho", type=float, default=0)
    parser.add_argument("--prior_sig", type=float, default=0.1)
    #HMC HPs
    parser.add_argument("--nr_chains", type=int)
    parser.add_argument("--step_size", type=float)
    parser.add_argument("--nr_samples", type=int)
    parser.add_argument("--L", type=int)
    parser.add_argument("--tau_out", type=float, default=1)
    parser.add_argument("--init", type=str, default='random')

    parser.add_argument("--save_model", type=bool)
    parser.add_argument("--evaluate_testset", type=bool)
    parser.add_argument("--evaluate_samples", type=bool, default = True)
    parser.add_argument("--use_nuts", type=bool, default = False)
    parser.add_argument("--tune_mm", type=bool, default = True)
    parser.add_argument("--device", type=str, default='gpu')
    parser.add_argument("--rep", type=int, default=0)

    config = load_config(config_file, config_name)
    parser.set_defaults(**config)
    args = parser.parse_args()

    return args