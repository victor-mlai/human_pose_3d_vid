import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Training script")

    # General arguments
    parser.add_argument(
        "-d",
        "--dataset",
        default="h36m",
        type=str,
        metavar="NAME",
        help="target dataset",
    )  # h36m or humaneva

    args = parser.parse_args()

    # Check invalid configuration
    # if args.resume and args.evaluate:
    #    print('Invalid flags: --resume and --evaluate cannot be set at the same time')
    #    exit()
    #
    # if args.export_training_curves and args.no_eval:
    #    print('Invalid flags: --export-training-curves and --no-eval cannot be set at the same time')
    #    exit()

    return args
