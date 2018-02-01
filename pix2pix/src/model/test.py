import os
import sys
import time
import numpy as np
import models
from keras.utils import generic_utils
from keras.optimizers import Adam, SGD
import keras.backend as K
# Utils
sys.path.append("../utils")
import general_utils
import data_utils



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('patch_size', type=int, nargs=2, action="store", help="Patch size for D")
    parser.add_argument('--backend', type=str, default="theano", help="theano or tensorflow")
    parser.add_argument('--generator', type=str, default="upsampling", help="upsampling or deconv")
    parser.add_argument('--dset', type=str, default="facades", help="facades")
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('--n_batch_per_epoch', default=100, type=int, help="Number of training epochs")
    parser.add_argument('--nb_epoch', default=400, type=int, help="Number of batches per epoch")
    parser.add_argument('--epoch', default=10, type=int, help="Epoch at which weights were saved for evaluation")
    parser.add_argument('--nb_classes', default=2, type=int, help="Number of classes")
    parser.add_argument('--do_plot', action="store_true", help="Debugging plot")
    parser.add_argument('--bn_mode', default=2, type=int, help="Batch norm mode")
    parser.add_argument('--img_dim', default=64, type=int, help="Image width == height")
    parser.add_argument('--use_mbd', action="store_true", help="Whether to use minibatch discrimination")
    parser.add_argument('--use_label_smoothing', action="store_true", help="Whether to smooth the positive labels when training D")
    parser.add_argument('--label_flipping', default=0, type=float, help="Probability (0 to 1.) to flip the labels when training D")

    args = parser.parse_args()

    # Set the backend by modifying the env variable
    if args.backend == "theano":
        os.environ["KERAS_BACKEND"] = "theano"
    elif args.backend == "tensorflow":
        os.environ["KERAS_BACKEND"] = "tensorflow"

    # Import the backend
    import keras.backend as K

    # manually set dim ordering otherwise it is not changed
    if args.backend == "theano":
        image_data_format = "channels_first"
        K.set_image_data_format(image_data_format)
    elif args.backend == "tensorflow":
        image_data_format = "channels_last"
        K.set_image_data_format(image_data_format)

    generator_model = models.load("generator_unet_%s" % generator,
                                img_dim,
                                nb_patch,
                                bn_mode,
                                use_mbd,
                                batch_size)