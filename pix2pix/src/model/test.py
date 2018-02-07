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
import argparse
import cv2
import imageio

def load_test_image(img_path, size, nb_channels):
    """
    Load img with opencv and reshape
    """

    if nb_channels == 1:
        img = cv2.imread(img_path, 0)
        img = np.expand_dims(img, axis=-1)
    else:
        img = cv2.imread(img_path)
        img = img[:, :, ::-1]  # GBR to RGB

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    if nb_channels == 1:
        img = np.expand_dims(img, -1)

    img = np.expand_dims(img, 0)

    return img

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
    parser.add_argument('--input_dir', type=str, help="Input directory")
    parser.add_argument('--output_dir', type=str, help="Output directory")
    parser.add_argument('--nb_channels', type=int, help="Number of channels")

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

    #img_dim = (args.img_dim, args.img_dim);
    print(args.patch_size)
    #X_full_train, X_sketch_train, X_full_val, X_sketch_val = data_utils.load_data(args.dset, image_data_format)
    #img_dim = X_full_train.shape[-3:]
    #print("!!!IMG_DIM:" + str(img_dim))
    img_dim = (args.img_dim*4,args.img_dim*4,3)

    nb_patch, img_dim_disc = data_utils.get_nb_patch(img_dim, args.patch_size, image_data_format)
    """
    generator_model = models.load("generator_unet_%s" % args.generator,
                                (None, None, 3),
                                nb_patch,
                                args.bn_mode,
                                args.use_mbd,
                                args.batch_size)"""
    generator_model = models.generator_unet_upsampling_fixdepth((None, None, 3), 
                                        args.bn_mode, 256, model_name="generator_unet_upsampling")
    
    #e = 10
    gen_weights_path = os.path.join('../../models/%s/gen_weights_epoch%s.h5' % ("CNN", args.epoch))
    generator_model.load_weights(gen_weights_path);

    if len(args.input_dir) != 0:
        files=os.listdir(args.input_dir)
        print("FILES:" + str(files))
        for f in files:
            print(f)
            img_test = load_test_image(args.input_dir + '/' + f, args.img_dim, args.nb_channels)
            result = generator_model.predict(data_utils.normalization(img_test))[0,:,:,:]
            
            result = data_utils.inverse_normalization(result) * 255
            result = result.astype(np.uint8)
            imageio.imwrite(args.output_dir + '/' + f, result)

