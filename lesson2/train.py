import os
import argparse
import json
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger, TensorBoard
from keras import backend as K
from .dataset import create_generators
from .dilated_unet import build_dilated_unet

BASEPATH = os.path.dirname(os.path.abspath(__file__))

def dice(y_true, y_pred):
    y_true_int = K.round(y_true)
    y_pred_int = K.round(y_pred)
    axis = [1, 2]
    intersection = K.sum(y_true_int * y_pred_int, axis=axis)
    area_true = K.sum(y_true_int, axis=axis)
    area_pred = K.sum(y_pred_int, axis=axis)
    batch_dice_coefs = (2 * intersection + 1) / (area_true + area_pred + 1)
    return K.mean(batch_dice_coefs, axis=0)

def iou(y_true, y_pred):
    y_true_int = K.round(y_true)
    y_pred_int = K.round(y_pred)
    axis = [1, 2]
    intersection = K.sum(y_true_int * y_pred_int, axis=axis)
    area_true = K.sum(y_true_int, axis=axis)
    area_pred = K.sum(y_true_int, axis=axis)
    union = area_true + area_pred - intersection
    batch_iou_scores = (intersection + 1) / (union + 1)
    return K.mean(batch_iou_scores, axis=0)

def train(config, name, images, masks):
    keys = ['rotation_range', 'width_shift_range', 'height_shift_range',
            'shear_range', 'zoom_range', 'fill_mode', 'alpha', 'sigma']
    augmentation_args = {k: v for k, v in config.items() if k in keys}

    keys = ['batch_size', 'validation_split', 'img_size', 'shuffle_train_val', 'shuffle',
            'seed', 'normalize_images', 'augment_training', 'augment_validation']
    create_generators_kwargs = {k: v for k, v in config.items() if k in keys}

    train_generator, train_steps_per_epoch, val_generator, val_steps_per_epoch = create_generators(images, masks, augmentation_args=augmentation_args, **create_generators_kwargs
    )

    # get image dimensions from first batch
    images, masks = next(train_generator)
    _, height, width, channels = images.shape

    keys = ['features', 'depth', 'padding', 'batchnorm', 'dilation_layers']
    build_dilated_unet_kwargs = {k: v for k, v in config.items() if k in keys}
    model = build_dilated_unet(height, width, channels, **build_dilated_unet_kwargs)



    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', dice, iou])

    trained_fp = os.path.join(BASEPATH, f'../trained/model_{name}.hdf5')
    log_fp = os.path.join(BASEPATH, f'../logs/model_{name}.log')
    tensorboard_log_dir = os.path.join(BASEPATH, f'../logs/tensorboard/model_{name}')
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, verbose=1, min_delta=1e-4),
        # ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1, epsilon=1e-4),
        ModelCheckpoint(trained_fp, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
        CSVLogger(log_fp),
        TensorBoard(log_dir=tensorboard_log_dir)
    ]

    history = model.fit_generator(train_generator, epochs=20,
                        steps_per_epoch=train_steps_per_epoch,
                        validation_data=val_generator,
                        validation_steps=val_steps_per_epoch,
                        callbacks=callbacks,
                        verbose=2)
    
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config file path')
    args = parser.parse_args()

    name = os.path.basename(args.config).split('.')[0]

    with open(args.config, 'r') as f:
        config = json.load(f)

    train(config, name)
