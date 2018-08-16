import os
import random
import numpy as np
from math import ceil
from scipy.misc import imresize
from skimage import io
from scipy.ndimage import map_coordinates, gaussian_filter
from keras.preprocessing.image import ImageDataGenerator
import mdai
import cv2

def load_images(imgs_anns_dict, img_size = 128): 
    #images = np.zeros((img_height, img_width, count), dtype=np.uint8)
    #mask = np.zeros((img_height, img_width, count), dtype=np.uint8)
    images = []
    masks = []
    
    for img_fp, ann in imgs_anns_dict.items():
        img = mdai.visualize.load_dicom_image(img_fp)
   
        img_width = img.shape[1]
        img_height = img.shape[0]
    
        mask = np.zeros((img_height, img_width), dtype=np.uint8) 
        
        assert img.shape == mask.shape
        
        for a in ann:     
            vertices = np.array(a['data']['vertices'])
            vertices = vertices.reshape((-1,2))                     
            cv2.fillPoly(mask, np.int32([vertices]), (255,255,255))
        
        # resize
        if img.shape[0] == img.shape[1]:
            resized_shape = (img_size, img_size)
            offset = (0, 0)
        elif img.shape[0] > img.shape[1]:
            resized_shape = (img_size, round(img_size * img.shape[1] / img.shape[0]))
            offset = (0, (img_size - resized_shape[1]) // 2)
        else:
            resized_shape = (round(img_size * img.shape[0] / img.shape[1]), img_size)
            offset = ((img_size - resized_shape[0]) // 2, 0)
            
        resized_shape_cv2_convention = (resized_shape[1], resized_shape[0])
        img_resized = cv2.resize(img, resized_shape_cv2_convention).astype(np.uint8)
        mask_resized = cv2.resize(mask, resized_shape_cv2_convention).astype(np.bool)

        # add padding to square
        img_padded = np.zeros((img_size, img_size), dtype=np.uint8)
        img_padded[offset[0]:(offset[0] + resized_shape[0]), offset[1]:(offset[1] + resized_shape[1])] = img_resized
        mask_padded = np.zeros((img_size, img_size), dtype=np.bool)
        mask_padded[offset[0]:(offset[0] + resized_shape[0]), offset[1]:(offset[1] + resized_shape[1])] = mask_resized

        images.append(img_padded)
        masks.append(mask_padded)

    # add channel dim
    images = np.asarray(images)[:, :, :, None]
    masks = np.asarray(masks)[:, :, :, None]
    return images, masks


# def load_images(img_size=256):
#     """Loads images and masks into memory.
#     """
#     images = []
#     masks = []

#     for image_fp, left_lung_mask_fp, right_lung_mask_fp in get_filepath_tuples_with_masks():
#         img = io.imread(image_fp)
#         left_lung_mask = io.imread(left_lung_mask_fp).astype(np.bool)
#         right_lung_mask = io.imread(right_lung_mask_fp).astype(np.bool)
#         mask = np.logical_or(left_lung_mask, right_lung_mask)

#         assert img.shape == mask.shape

#         # resize
#         if img.shape[0] == img.shape[1]:
#             resized_shape = (img_size, img_size)
#             offset = (0, 0)
#         elif img.shape[0] > img.shape[1]:
#             resized_shape = (img_size, round(img_size * img.shape[1] / img.shape[0]))
#             offset = (0, (img_size - resized_shape[1]) // 2)
#         else:
#             resized_shape = (round(img_size * img.shape[0] / img.shape[1]), img_size)
#             offset = ((img_size - resized_shape[0]) // 2, 0)
#         img_resized = imresize(img, resized_shape, interp='lanczos').astype(np.uint8)
#         mask_resized = imresize(mask, resized_shape, interp='lanczos').astype(np.bool)

#         # add padding to square
#         img_padded = np.zeros((img_size, img_size), dtype=np.uint8)
#         img_padded[offset[0]:(offset[0] + resized_shape[0]), offset[1]:(offset[1] + resized_shape[1])] = img_resized
#         mask_padded = np.zeros((img_size, img_size), dtype=np.bool)
#         mask_padded[offset[0]:(offset[0] + resized_shape[0]), offset[1]:(offset[1] + resized_shape[1])] = mask_resized

#         images.append(img_padded)
#         masks.append(mask_padded)

#     # add channel dim
#     images = np.asarray(images)[:, :, :, None]
#     masks = np.asarray(masks)[:, :, :, None]

#     return images, masks


def random_elastic_deformation(image, alpha, sigma, mode='nearest', random_state=None):
    """Elastic deformation of images as described in: Simard, Steinkraus and Platt, "Best Practices for Convolutional
    Neural Networks applied to Visual Document Analysis", in Proc. of the International Conference on Document Analysis
    and Recognition, 2003.
    """
    assert len(image.shape) == 3

    if random_state is None:
        random_state = np.random.RandomState(None)

    height, width, channels = image.shape

    dx = gaussian_filter(2 * random_state.rand(height, width) - 1, sigma, mode='constant', cval=0) * alpha
    dy = gaussian_filter(2 * random_state.rand(height, width) - 1, sigma, mode='constant', cval=0) * alpha

    x, y = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    indices = (np.repeat(np.ravel(x + dx), channels),
               np.repeat(np.ravel(y + dy), channels),
               np.tile(np.arange(channels), height * width))

    values = map_coordinates(image, indices, order=1, mode=mode)

    return values.reshape((height, width, channels))


def normalize(x, epsilon=1e-7, axis=(1, 2)):
    x -= np.mean(x, axis=axis, keepdims=True)
    x /= (np.std(x, axis=axis, keepdims=True) + epsilon)


class Iterator(object):
    def __init__(self, images, masks, batch_size,
                 shuffle=True,
                 rotation_range=180,
                 width_shift_range=0.3,
                 height_shift_range=0.1,
                 shear_range=0.1,
                 zoom_range=0.1,
                 fill_mode='nearest',
                 alpha=500,
                 sigma=20):
        self.images = images
        self.masks = masks
        self.batch_size = batch_size
        self.shuffle = shuffle
        augment_options = {
            'rotation_range': rotation_range,
            'width_shift_range': width_shift_range,
            'height_shift_range': height_shift_range,
            'shear_range': shear_range,
            'zoom_range': zoom_range,
            'fill_mode': fill_mode,
        }
        self.idg = ImageDataGenerator(**augment_options)
        self.alpha = alpha
        self.sigma = sigma
        self.fill_mode = fill_mode
        self.i = 0
        self.index = np.arange(len(images))
        if shuffle:
            np.random.shuffle(self.index)

    def __next__(self):
        return self.next()

    def next(self):
        # compute how many images to output in this batch
        start = self.i
        end = min(start + self.batch_size, len(self.images))

        augmented_images = []
        augmented_masks = []
        for n in self.index[start:end]:
            image = self.images[n]
            mask = self.masks[n]

            channels = image.shape[2]

            # stack image + mask together to simultaneously augment
            stacked = np.concatenate((image, mask), axis=2)

            # apply simple affine transforms first using Keras
            augmented = self.idg.random_transform(stacked)

            # maybe apply elastic deformation
            if self.alpha != 0 and self.sigma != 0:
                augmented = random_elastic_deformation(augmented, self.alpha, self.sigma, self.fill_mode)

            # split image and mask back apart
            augmented_image = augmented[:, :, :channels]
            augmented_images.append(augmented_image)
            augmented_mask = np.round(augmented[:, :, channels:])
            augmented_masks.append(augmented_mask)

        self.i += self.batch_size
        if self.i >= len(self.images):
            self.i = 0
            if self.shuffle:
                np.random.shuffle(self.index)

        return np.asarray(augmented_images), np.asarray(augmented_masks)


def create_generators(images, masks, batch_size=16, validation_split=0.0, img_size=256,
                      shuffle_train_val=True, shuffle=True, seed=None,
                      normalize_images=True, augment_training=False,
                      augment_validation=False, augmentation_args={}):
    """Creates generators.
    """
    #images, masks = load_images(img_size=img_size)

    # cast as float64
    images = images.astype(np.float64)

    # maybe normalize image
    if normalize_images:
        normalize(images, axis=(1, 2))

    if seed is not None:
        np.random.seed(seed)

    if shuffle_train_val:
        # shuffle images and masks in parallel
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(masks)

    # split out last %(validation_split) of images as validation set
    split_index = int((1 - validation_split) * len(images))

    if augment_training:
        train_generator = Iterator(images[:split_index], masks[:split_index], batch_size,
                                   shuffle=shuffle, **augmentation_args)
    else:
        idg = ImageDataGenerator()
        train_generator = idg.flow(images[:split_index], masks[:split_index], batch_size=batch_size, shuffle=shuffle)

    train_steps_per_epoch = ceil(split_index / batch_size)

    if validation_split > 0.0:
        if augment_validation:
            val_generator = Iterator(images[split_index:], masks[split_index:], batch_size,
                                     shuffle=shuffle, **augmentation_args)
        else:
            idg = ImageDataGenerator()
            val_generator = idg.flow(images[split_index:], masks[split_index:], batch_size=batch_size, shuffle=shuffle)
    else:
        val_generator = None

    val_steps_per_epoch = ceil((len(images) - split_index) / batch_size)

    return (train_generator, train_steps_per_epoch, val_generator, val_steps_per_epoch)
