import argparse
import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--cpu', action='store_true')

    parser.add_argument(
        '--output_dir', 
        type=str, 
        default=None,
        help='Output dir for this experiment'
    )

    parser.add_argument(
        '--dataset', 
        type=str, 
        default='fashionmnist',
        help='Can be fhashionmnist or cifar10'
    )

    parser.add_argument(
        '--resnet_version',
        type=str,
        default='101',
        help='Which resnet version to use'
    )

    parser.add_argument(
        '--with_deformable_conv',
        action='store_true'
    )
    
    # By default replace all 3*3 cpmv filter with deformconv2d in conv4 and conv5
    parser.add_argument('--dcn', nargs=4, type=bool, default=[False, False, True, True])

    # With early stopping
    parser.add_argument('--early_stopping', action='store_true')

    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="patience for early stopping",
    )
    parser.add_argument(
        "--mode", type=str, default='min', help="optimise towards minimising or maximising loss function"
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=0.0005,
        help="minimum difference in performance for early stopping",
    )

    parser.add_argument('--restore_best_weights', action='store_true')

    # Learning rate
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0005,
        help="Learning Rate",
    )
    # Train batch size
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="Batch size for the training dataloader."
    )

    # Batch size for both test and val dataloader
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Batch size for the evaluation dataloader.",
    )

    parser.add_argument("--num_epochs", type=int, default=10, help="Total number of training epochs to perform.")

    args = parser.parse_args()

    return args