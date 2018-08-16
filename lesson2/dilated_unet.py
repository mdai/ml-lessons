from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D
from keras.layers import Cropping2D, Concatenate, BatchNormalization, Activation
from keras.models import Model
from keras import backend as K


def downsampling_block(input_tensor, filters, padding='same', batchnorm=False):
    _, height, width, _ = K.int_shape(input_tensor)
    assert height % 2 == 0
    assert width % 2 == 0

    x = Conv2D(filters, kernel_size=(3, 3), padding=padding, dilation_rate=1)(input_tensor)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size=(3, 3), padding=padding, dilation_rate=2)(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)

    return MaxPooling2D(pool_size=(2, 2))(x), x


def upsampling_block(input_tensor, skip_tensor, filters, padding='same', batchnorm=False):
    x = Conv2DTranspose(filters, kernel_size=(2, 2), strides=(2, 2))(input_tensor)

    # compute amount of cropping needed for skip_tensor
    _, x_height, x_width, _ = K.int_shape(x)
    _, s_height, s_width, _ = K.int_shape(skip_tensor)
    h_crop = s_height - x_height
    w_crop = s_width - x_width
    assert h_crop >= 0
    assert w_crop >= 0
    if h_crop == 0 and w_crop == 0:
        y = skip_tensor
    else:
        cropping = ((h_crop // 2, h_crop - h_crop // 2), (w_crop // 2, w_crop - w_crop // 2))
        y = Cropping2D(cropping=cropping)(skip_tensor)

    x = Concatenate()([x, y])

    # no dilation in upsampling convolutions
    x = Conv2D(filters, kernel_size=(3, 3), padding=padding)(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size=(3, 3), padding=padding)(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)

    return x


def build_dilated_unet(height, width, channels,
                       features=64, depth=4, padding='same', batchnorm=False, dilation_layers=5):
    """Generate dilated U-Net model where the convolutions in the encoding and bottleneck are replaced by dilated
    convolutions. The second convolution in pair at a given scale in the encoder is dilated by 2. The number of dilation
    layers in the innermost bottleneck is controlled by the `dilation_layers` parameter -- this is the `context module`
    proposed by Yu, Koltun 2016 in "Multi-scale Context Aggregation by Dilated Convolutions". Number of features double
    after each down sampling block.

    Arguments:
      height - input image height (pixels)
      width - input image width (pixels)
      channels - input image features (1 for grayscale, 3 for RGB)
      features - number of output features for first convolution (64 in paper)
      depth - number of downsampling operations (4 in paper)
      padding - 'valid' (used in paper) or 'same'
      batchnorm - include batch normalization layers before activations
      dilation_layers - number of dilated convolutions in innermost bottleneck

    Output:
      Dilated U-Net model expecting input shape (height, width, maps) and generates output with shape (output_height,
      output_width, 1). If padding is 'same', then output_height = height and output_width = width.
    """

    x = Input(shape=(height, width, channels))
    inputs = x

    skips = []
    for i in range(depth):
        x, x0 = downsampling_block(x, features, padding, batchnorm)
        skips.append(x0)
        features *= 2

    dilation_rate = 1
    for n in range(dilation_layers):
        x = Conv2D(filters=features, kernel_size=(3, 3), padding=padding, dilation_rate=dilation_rate)(x)
        x = BatchNormalization()(x) if batchnorm else x
        x = Activation('relu')(x)
        dilation_rate *= 2

    for i in reversed(range(depth)):
        features //= 2
        x = upsampling_block(x, skips[i], features, padding, batchnorm)

    x = Conv2D(filters=1, kernel_size=(1, 1))(x)

    probabilities = Activation('sigmoid')(x)

    return Model(inputs=inputs, outputs=probabilities)
