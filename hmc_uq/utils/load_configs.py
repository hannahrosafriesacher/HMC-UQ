import argparse
import yaml

def load_config(config_file, config_name="default"):
    """Loads a specific configuration from a YAML file."""
    with open(config_file, "r") as f:
        all_configs = yaml.safe_load(f)
    
    if config_name not in all_configs:
        raise ValueError(f"Config '{config_name}' not found in {config_file}. Available configs: {list(all_configs.keys())}")
    
    return all_configs[config_name]

def get_args():
    """Parses command-line arguments, with defaults from a YAML file."""
    parser = argparse.ArgumentParser(description="Model Configs")

    # Add the option to load from YAML
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--config_name", type=str, default="default", help="Which config to use from the YAML file")

    # Define model arguments
    parser.add_argument("--TargetID", type=int, default=None)
    parser.add_argument("--nr_layers", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--hidden_sizes", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--tr_fold", type=list, default=[2, 3, 4])
    parser.add_argument("--va_fold", type=int, default=1)
    parser.add_argument("--te_fold", type=int, default=0)
    parser.add_argument("--model_loss", type=str, default="BCEwithlogitsloss")
    parser.add_argument("--evaluate_testset", type=bool, default=True)
    parser.add_argument("--save_model", type=bool, default=False)

    args = parser.parse_args()

    # If a config file is provided, load the selected configuration
    if args.config:
        config = load_config(args.config, args.config_name)
        for key, value in config.items():
            if getattr(args, key) is None:  # Only override if not provided in CLI
                setattr(args, key, value)

    return args