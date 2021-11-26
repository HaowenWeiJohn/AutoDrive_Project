from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.backend import concatenate
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, TimeDistributed, ConvLSTM2D, \
    BatchNormalization, AveragePooling2D
import tensorflow as tf

# input size (w,h,n)    n is back track images at time t, in this case we deal it as channel
from tensorflow.python.layers.convolutional import Conv2DTranspose

from model.ResLSTM.loss import Custom_loss
from model.ResLSTM.metrics import MOS_IoU


def time_dis_conv2d(input, filters=64 ,kernal_size=(3,3), dilation_rate=(1,1), padding='same',
                               activation=tf.keras.layers.LeakyReLU(alpha=0.1), batch_norm=True):

    output = TimeDistributed(Conv2D(filters=filters, kernel_size=kernal_size,
                                  # activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                  # kernel_regularizer=tf.keras.regularizers.l2(l=1e-5),
                                  # bias_regularizer=tf.keras.regularizers.l2(l=1e-5),
                                  # activity_regularizer=tf.keras.regularizers.l2(l=1e-5),
                                   dilation_rate=dilation_rate,
                                   padding=padding))(input)

     # activation function
    if activation:
        output = TimeDistributed(activation)(output)

    if batch_norm:
        output = TimeDistributed(BatchNormalization())(output)

    return output


def conv2d_layer(input, filters=64 ,kernal_size=(3,3), dilation_rate=(1,1), padding='same',
                               activation=tf.keras.layers.LeakyReLU(alpha=0.1), batch_norm=True):

    output = Conv2D(filters=filters, kernel_size=kernal_size,
                                  # activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                  # kernel_regularizer=tf.keras.regularizers.l2(l=1e-5),
                                  # bias_regularizer=tf.keras.regularizers.l2(l=1e-5),
                                  # activity_regularizer=tf.keras.regularizers.l2(l=1e-5),
                                   dilation_rate=dilation_rate,
                                   padding=padding)(input)

     # activation function
    if activation:
        output = activation(output)

    if batch_norm:
        output = BatchNormalization()(output)

    return output




def time_dis_conv2d_transpose(input, filters=64 ,kernal_size=(3,3), dilation_rate=(1,1), stride=(2,2), padding='same',
                               activation=tf.keras.layers.LeakyReLU(alpha=0.1), batch_norm=True):

    output = TimeDistributed(Conv2DTranspose(filters=filters, kernel_size=kernal_size, strides=stride,
                                  # activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                  # kernel_regularizer=tf.keras.regularizers.l2(l=1e-5),
                                  # bias_regularizer=tf.keras.regularizers.l2(l=1e-5),
                                  # activity_regularizer=tf.keras.regularizers.l2(l=1e-5),
                                   dilation_rate=dilation_rate,
                                   padding=padding))(input)

     # activation function
    if activation:
        output = TimeDistributed(activation)(output)

    if batch_norm:
        output = TimeDistributed(BatchNormalization())(output)

    return output



def conv_lstm2d(input, filters=64 ,kernal_size=(3,3), dilation_rate=(1,1), padding='same',
                               activation=tf.keras.layers.LeakyReLU(alpha=0.1), return_sequence=True, batch_norm=True):

    output = ConvLSTM2D(filters=filters, kernel_size=kernal_size,
                                  # activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                  # kernel_regularizer=tf.keras.regularizers.l2(l=1e-5),
                                  # bias_regularizer=tf.keras.regularizers.l2(l=1e-5),
                                  # activity_regularizer=tf.keras.regularizers.l2(l=1e-5),
                                   dilation_rate=dilation_rate,
                                   padding=padding,
                                   return_sequences=return_sequence)(input)

     # activation function
    if activation:
        output = TimeDistributed(activation)(output)

    if batch_norm:
        output = TimeDistributed(BatchNormalization())(output)

    return output


def conv_lstm2d_upsampling(input, filters=64 ,kernal_size=(3,3), dilation_rate=(1,1), padding='same', upsampling_size=(2,2),
                               activation=tf.keras.layers.LeakyReLU(alpha=0.1), return_sequence=True, batch_norm=True):

    output = ConvLSTM2D(filters=filters, kernel_size=kernal_size,
                                  # activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                  # kernel_regularizer=tf.keras.regularizers.l2(l=1e-5),
                                  # bias_regularizer=tf.keras.regularizers.l2(l=1e-5),
                                  # activity_regularizer=tf.keras.regularizers.l2(l=1e-5),
                                   dilation_rate=dilation_rate,
                                   padding=padding,
                                   return_sequences=return_sequence)(UpSampling2D(size=upsampling_size)(input))

     # activation function
    if activation:
        output = TimeDistributed(activation)(output)

    if batch_norm:
        output = TimeDistributed(BatchNormalization())(output)

    return output




def lstm_dilation_block(input, return_sequence=True):
    # conv, conv, conv_concad, dilation
    input_shape = input.shape
    input_channel = input_shape[-1]

    conv_lstm_1 = conv_lstm2d(input, filters=input_channel, kernal_size=(1,1), dilation_rate=(1,1), padding='same',
                               activation=tf.keras.layers.LeakyReLU(alpha=0.1), return_sequence=True, batch_norm=True)

    conv_lstm_2 = conv_lstm2d(conv_lstm_1, filters=input_channel, kernal_size=(3,3), dilation_rate=(1,1), padding='same',
                               activation=tf.keras.layers.LeakyReLU(alpha=0.1), return_sequence=True, batch_norm=True)

    conv_lstm_3 = conv_lstm2d(conv_lstm_1, filters=input_channel, kernal_size=(3,3), dilation_rate=(2,2), padding='same',
                               activation=tf.keras.layers.LeakyReLU(alpha=0.1), return_sequence=True, batch_norm=True)

    concatenate_layer = concatenate([conv_lstm_1, conv_lstm_2,conv_lstm_3])

    conv_lstm_4 = conv_lstm2d(concatenate_layer, filters=input_channel, kernal_size=(1, 1), dilation_rate=(1, 1),
                              padding='same',
                              activation=tf.keras.layers.LeakyReLU(alpha=0.1), return_sequence=return_sequence, batch_norm=True)
    output = conv_lstm_4

    return output

def time_dis_down_sampling(input, dropout_rate=0.5, pool_size=(2,2), stride=(2, 2), padding='same'):

    output = Dropout(rate=dropout_rate)(input)
    output = TimeDistributed(MaxPooling2D(pool_size=pool_size, strides=stride, padding=padding))(output)

    return output


def time_dis_dilation_down(input):

    input_shape = input.shape
    input_channel = input_shape[-1]
    conv_1 = time_dis_conv2d(input=input, filters=input_channel, kernal_size=(1, 1), dilation_rate=(1, 1), padding='same',
                                activation=tf.keras.layers.LeakyReLU(alpha=0.1), batch_norm=False)

    conv_2 = time_dis_conv2d(input=conv_1, filters=input_channel, kernal_size=(3, 3), dilation_rate=(1, 1), padding='same',
                                activation=tf.keras.layers.LeakyReLU(alpha=0.1), batch_norm=True)

    conv_3 = time_dis_conv2d(input=conv_2, filters=input_channel, kernal_size=(2, 2), dilation_rate=(2, 2), padding='same',
                                activation=tf.keras.layers.LeakyReLU(alpha=0.1), batch_norm=True)

    mid = concatenate([conv_2, conv_3])

    conv_4 = time_dis_conv2d(input=mid, filters=mid.shape[-1], kernal_size=(1, 1), dilation_rate=(1, 1), padding='same',
                                activation=tf.keras.layers.LeakyReLU(alpha=0.1), batch_norm=True)

    return conv_4

def time_dis_dilation_up(input):

    input_shape = input.shape
    input_channel = input_shape[-1]
    conv_1 = time_dis_conv2d(input=input, filters=input_channel, kernal_size=(1, 1), dilation_rate=(1, 1), padding='same',
                                activation=tf.keras.layers.LeakyReLU(alpha=0.1), batch_norm=False)

    conv_2 = time_dis_conv2d(input=conv_1, filters=input_channel, kernal_size=(3, 3), dilation_rate=(1, 1), padding='same',
                                activation=tf.keras.layers.LeakyReLU(alpha=0.1), batch_norm=True)

    conv_3 = time_dis_conv2d(input=conv_2, filters=input_channel, kernal_size=(3, 3), dilation_rate=(2, 2), padding='same',
                                activation=tf.keras.layers.LeakyReLU(alpha=0.1), batch_norm=True)

    mid = concatenate([conv_2, conv_3])

    conv_4 = time_dis_conv2d(input=mid, filters=input_channel/2, kernal_size=(1, 1), dilation_rate=(1, 1), padding='same',
                                activation=tf.keras.layers.LeakyReLU(alpha=0.1), batch_norm=True)

    return conv_4


####################################################################
def context_block(input):
    conv1 = time_dis_conv2d(input=input, filters=32,kernal_size=(1,1), dilation_rate=(1,1), padding='same',
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.1), batch_norm=False)
    conv_2 = time_dis_conv2d(input=conv1, filters=32,kernal_size=(3,3), dilation_rate=(1,1), padding='same',
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.1), batch_norm=True)
    conv_3 = time_dis_conv2d(input=conv_2, filters=32,kernal_size=(3,3), dilation_rate=(2,2), padding='same',
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.1), batch_norm=True)

    output = conv1+conv_3

    return output




def dilation_down_block(input, filters=128):

    conv_1 = conv2d_layer(input, filters=filters,kernal_size=(1,1), dilation_rate=(1,1), padding='same',
                               activation=tf.keras.layers.LeakyReLU(alpha=0.1), batch_norm=False)


    conv_2 = conv2d_layer(conv_1, filters=filters, kernal_size=(3, 3), dilation_rate=(1, 1), padding='same',
                          activation=tf.keras.layers.LeakyReLU(alpha=0.1), batch_norm=True)

    conv_3 = conv2d_layer(conv_2, filters=filters, kernal_size=(3, 3), dilation_rate=(2, 2), padding='same',
                          activation=tf.keras.layers.LeakyReLU(alpha=0.1), batch_norm=True)

    conv_4 = conv2d_layer(conv_3, filters=filters, kernal_size=(2, 2), dilation_rate=(2, 2), padding='same',
                          activation=tf.keras.layers.LeakyReLU(alpha=0.1), batch_norm=True)

    concatenate_layer = concatenate([conv_2, conv_3, conv_4])

    conv_5 = conv2d_layer(concatenate_layer, filters=filters, kernal_size=(1, 1), dilation_rate=(1, 1), padding='same',
                          activation=tf.keras.layers.LeakyReLU(alpha=0.1), batch_norm=True)

    output = conv_1+conv_5

    return output


def dilation_up_block(input, filters=128):
    conv_1 = conv2d_layer(input, filters=filters, kernal_size=(3, 3), dilation_rate=(1, 1), padding='same',
                          activation=tf.keras.layers.LeakyReLU(alpha=0.1), batch_norm=True)

    conv_2 = conv2d_layer(conv_1, filters=filters, kernal_size=(3, 3), dilation_rate=(1, 1), padding='same',
                          activation=tf.keras.layers.LeakyReLU(alpha=0.1), batch_norm=True)

    conv_3 = conv2d_layer(conv_2, filters=filters, kernal_size=(3, 3), dilation_rate=(2, 2), padding='same',
                          activation=tf.keras.layers.LeakyReLU(alpha=0.1), batch_norm=True)

    concatenate_layer = concatenate([conv_1, conv_2, conv_3])

    conv_4 = conv2d_layer(concatenate_layer, filters=filters, kernal_size=(1, 1), dilation_rate=(2, 2), padding='same',
                          activation=tf.keras.layers.LeakyReLU(alpha=0.1), batch_norm=True)

    output = Dropout(rate=0.5)(conv_4)
    return output


def pixel_shuffle_up(input, block_size=2):
    up_pixel_shuffle = tf.nn.depth_to_space(input, block_size=block_size)
    output = Dropout(rate=0.5)(up_pixel_shuffle)
    return output

def res_concatenate(input_1, input_2, dropout_rate=0.5):
    output = concatenate([input_1, input_2])
    output = Dropout(dropout_rate)(output)
    return output

####################################################################

def down_max_pool_block(input, dropout_rate=0.5, pool_size=(2,2), stride=(2, 2), padding='same'):

    output = Dropout(rate=dropout_rate)(input)
    output = MaxPooling2D(pool_size=pool_size, strides=stride, padding=padding)(output)

    return output


def down_average_pool_block(input, dropout_rate=0.5, pool_size=(2,2), stride=(2, 2), padding='same'):

    output = Dropout(rate=dropout_rate)(input)
    output = AveragePooling2D(pool_size=pool_size, strides=stride, padding=padding)(output)

    return output

########################################################
#  (r,x,y,z,e, res1, res2, res3)
def ResLSTM(input_shape = (5, 64, 2048, 9)):
    input_size = input_shape


    input = tf.keras.Input(shape=input_size)

    # context_1 = time_dis_conv2d(input=input, filters=32,kernal_size=(1,1), dilation_rate=(1,1), padding='same',
    #                                activation=tf.keras.layers.LeakyReLU(alpha=0.1), batch_norm=True)
    # context_2 = time_dis_conv2d(input=context_1, filters=32,kernal_size=(3,3), dilation_rate=(1,1), padding='same',
    #                                activation=tf.keras.layers.LeakyReLU(alpha=0.1), batch_norm=True)
    # context_3 = time_dis_conv2d(input=context_1, filters=32,kernal_size=(3,3), dilation_rate=(2,2), padding='same',
    #                                activation=tf.keras.layers.LeakyReLU(alpha=0.1), batch_norm=True)

    # context_concatenate = concatenate([context_1, context_2, context_3])

    context_out = context_block(input)


    # context_out = time_dis_conv2d(input=context_concatenate, filters=32,kernal_size=(1,1), dilation_rate=(1,1), padding='same',
    #                                activation=tf.keras.layers.LeakyReLU(alpha=0.1), batch_norm=True)

    dilation_lstm_1 = lstm_dilation_block(input=context_out, return_sequence=False)

    # merge_feature = concatenate([context_out[:,-1,:,:,:], dilation_lstm_1])

    dilation_down_1 = dilation_down_block(dilation_lstm_1, filters=32)
    down_sampling_1 = down_average_pool_block(dilation_down_1)

    dilation_down_2 = dilation_down_block(down_sampling_1, filters=64)
    down_sampling_2 = down_average_pool_block(dilation_down_2)

    dilation_down_3 = dilation_down_block(down_sampling_2, filters=128)
    down_sampling_3 = down_average_pool_block(dilation_down_3)

    dilation_down_4 = dilation_down_block(down_sampling_3, filters=128)
    down_sampling_4 = down_average_pool_block(dilation_down_4)


    dilation_down_5 = dilation_down_block(down_sampling_4, filters=256)

    pixel_shuffle_up_1 = pixel_shuffle_up(dilation_down_5)
    res_concatenate_1 = res_concatenate(dilation_down_4 ,pixel_shuffle_up_1)
    dilation_block_up_1 = dilation_up_block(res_concatenate_1, filters=128)

    pixel_shuffle_up_2 = pixel_shuffle_up(dilation_block_up_1)
    res_concatenate_2 = res_concatenate(dilation_down_3, pixel_shuffle_up_2)
    dilation_block_up_2 = dilation_up_block(res_concatenate_2, filters=128)

    pixel_shuffle_up_3 = pixel_shuffle_up(dilation_block_up_2)
    res_concatenate_3 = res_concatenate(dilation_down_2, pixel_shuffle_up_3)
    dilation_block_up_3 = dilation_up_block(res_concatenate_3, filters=64)

    pixel_shuffle_up_4 = pixel_shuffle_up(dilation_block_up_3)
    res_concatenate_4 = res_concatenate(dilation_down_1, pixel_shuffle_up_4)
    dilation_block_up_4 = dilation_up_block(res_concatenate_4, filters=32)

    dense_1 = conv2d_layer(dilation_block_up_4, filters=32, kernal_size=(1,1), dilation_rate=(1,1), padding='same',
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.1), batch_norm=True)

    dense_2 = conv2d_layer(dense_1, filters=16, kernal_size=(1,1), dilation_rate=(1,1), padding='same',
                                   activation=tf.keras.layers.LeakyReLU(alpha=0.1), batch_norm=True)

    # dense_3 = conv2d_layer(dense_2, filters=3, kernal_size=(1,1), dilation_rate=(1,1), padding='same',
    #                                activation=tf.keras.layers.LeakyReLU(alpha=0.1), batch_norm=True)

    logistic = Conv2D(filters=3, kernel_size=(1,1), activation='sigmoid')(dense_2)

    output = tf.keras.layers.Softmax(axis=-1)(logistic)

    model = Model(input, output)

    adam = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-5)
    #
    # tf.keras.losses.CategoricalCrossentropy(
    #     from_logits=False, label_smoothing=0.0, axis=-1,
    #     reduction=losses_utils.ReductionV2.AUTO,
    #     name='categorical_crossentropy'
    # )
    custom_loss = Custom_loss(class_weight=[0, 9, 251])
    mos_iou = MOS_IoU(name='mos_iou')
    model.compile(optimizer=adam, loss=custom_loss, metrics=[mos_iou])
    model.summary()
    return model

ResLSTM()





