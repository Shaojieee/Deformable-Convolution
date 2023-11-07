import argparse
import datetime



def main_parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--tune', action='store_true', help='To perform optuna model tuning for LR and optimiser')

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
        help='Can be fashionmnist or cifar10'
    )

    parser.add_argument(
        '--resnet_version',
        type=str,
        default='101',
        help='Which resnet version to use'
    )
    
    # Replace the `x` layer in each resnet block with deformable convolution
    parser.add_argument('--with_deformable_conv', nargs=4, type=int, default=[0,0,0,0])

    # Unfreezing the last `x` 3*3 conv layer in corresponding resnet block
    parser.add_argument('--unfreeze_conv', nargs=4, type=int, default=[0,0,0,0])

    parser.add_argument('--unfreeze_offset', action='store_true')
    parser.add_argument('--unfreeze_fc', action='store_true')

    parser.add_argument('--model_weights', type=str, default=None, help='File path to model weight')

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

    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    return args


def offset_parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--video', action='store_true')
    parser.add_argument(
        '--fps',
        type=int,
        help='FPS for the video'
    )
    parser.add_argument(
        '--duration',
        type=int,
        help='Duration for the video'
    )

    parser.add_argument(
        '--output_dir', 
        type=str,
        help='Output Dir'
    )

    parser.add_argument(
        '--image_file', 
        type=str,
        help='Image file to generate'
    )

    parser.add_argument(
        '--model_weights', 
        type=str,
        help='Model Weights '
    )


    parser.add_argument(
        '--resnet_version',
        type=str,
        default='101',
        help='Which resnet version to use'
    )
    
    # Replace the `x` layer in each resnet block with deformable convolution
    parser.add_argument('--with_deformable_conv', nargs=4, type=int, default=[0,0,0,0])

    # Unfreezing the last `x` 3*3 conv layer in corresponding resnet block
    parser.add_argument('--unfreeze_conv', nargs=4, type=int, default=[0,0,0,0])

    parser.add_argument('--unfreeze_offset', action='store_true')
    parser.add_argument('--unfreeze_fc', action='store_true')

    parser.add_argument('--num_classes', type=int, default=10)


    args = parser.parse_args()

    return args