'''
argparse options.
'''
import argparse
import yaml
import ast


def get_argparse_input():
    '''
    Adds general options to the given argparse parser.
    These are options that are shares across train, test, and visualization time.
    '''

    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Regular Config Files
    parser.add_argument('--config', type=str, required=True)

    # General Parameters
    parser.add_argument('--outdir', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weights', type=str)
    parser.add_argument('--optimizer-weights', type=str)
    parser.add_argument('--scheduler-weights', type=str)
    parser.add_argument('--_wandb', type=str, default='')

    parser.add_argument('--only_val', type=str)
    parser.add_argument('--only_viz', type=str)
    parser.add_argument('--dataset_source', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--save_every', type=int)
    parser.add_argument('--run_number', type=int)

    # Dataset arguments
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--data_generation.line_density', type=str)
    parser.add_argument('--data_generation.num_points', type=int)
    parser.add_argument('--data_generation.resolution', type=int)

    # Additional data feature inputs
    parser.add_argument('--use_curvature', type=str)
    parser.add_argument('--use_uv', action='store_true')
    parser.add_argument('--use_sparse_normals', action='store_true')
    parser.add_argument('--use_curve_idxs', action='store_true')

    # gradients estimate hyperparams
    parser.add_argument('--model.gradient_knn', type=int)
    parser.add_argument('--model.kernel_width', type=float)
    parser.add_argument('--model.hinge_reg', type=float)

    # Model arguments
    parser.add_argument('--model.type', type=str, help="Model backbone type. Other model args are provided dynamically")

    args, unknown_args = parser.parse_known_args()

    # format booleans
    if args.use_curvature is not None:
        args.use_curvature = True if args.use_curvature in ["True", "true"] else False
    if args.only_viz is not None:
        args.only_viz = True if args.only_viz in ["True", "true"] else False
    if args.only_val is not None:
        args.only_val = True if args.only_val in ["True", "true"] else False

    # format into config dictionary
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    for arg in vars(args):
        if getattr(args, arg) is not None:
            arg_keys = arg.split(".")
            val = getattr(args, arg)
            recursive_dict_update(config, arg_keys, val)

    # overwrite config arguments with any extra unknown arguments
    unknown_args = hacky_format_unknown_args(unknown_args)
    unified_config = overwrite_configfile_fields(config, unknown_args)

    return unified_config


def overwrite_configfile_fields(unified_config, extra_args):
    for arg, val in extra_args.items():
        arg_keys = arg.split(".")
        if type(val) == dict:  # occurs becasue wandb passes in whole "config" as arguments
            print("Passed in dictionary as argument. Ignoring it:")
            print(arg_keys)
            continue
        if arg_keys[0] == "_wandb" or arg_keys[0] == "model" or arg_keys[0] == "scheduler":
            continue
        _, success = recursive_dict_update(unified_config, arg_keys, val)
        if not success:
            print("[WARNING] Command Line Argument %s Did Not Match a Config Key!" % arg)

    return unified_config


def hacky_format_unknown_args(unknown):
    def format_value(val):
        # list check
        if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
            val = ast.literal_eval(val)

        # boolean check
        if val == "True":
            val = True
        elif val == "False":
            val = False

        # int / float check
        try:
            val = float(val)
        except (ValueError, TypeError):  # it's a string
            pass


        return val

    formatted = {}
    for arg in unknown:
        if arg.startswith(("--")):
            name, val = arg.split('=')
            name = name[2:]
            val = format_value(val)
            formatted[name] = val
    return formatted


def recursive_dict_update(d, u, val):
    if len(u) == 0:
        return
    k = u[0]
    if len(u) > 1:
        if k in d:
            d[k], flag = recursive_dict_update(d[k], u[1:], val)
    else:
        if k in d:
            val_type = type(d[k])
            d[k] = val_type(val)
            flag = True
        else:
            flag = False

    return d, flag