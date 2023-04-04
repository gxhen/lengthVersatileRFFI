from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import tensorflow as tf

from deep_learning_models import rff_transformer, lstm_model, FCN_model, gru_model
from deep_learning_models import TransformerEncoder, PositionEmbedding

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop, SGD, Adam

from tensorflow.keras.utils import to_categorical

from utils import load_single_file, shuffle, awgn, ch_ind_spectrogram, data_generator


def train(data_path, tx_range, model_type):

    [data_sf7, label_sf7] = load_single_file(data_path[0],
                                             'data',
                                             'label',
                                             tx_range,
                                             pkt_range=range(0, 2500))

    [data_sf8, label_sf8] = load_single_file(data_path[1],
                                             'data',
                                             'label',
                                             tx_range,
                                             pkt_range=range(0, 2500))

    [data_sf9, label_sf9] = load_single_file(data_path[2],
                                             'data',
                                             'label',
                                             tx_range,
                                             pkt_range=range(0, 2500))

    label_sf7 = to_categorical(label_sf7 - tx_range[0])
    label_sf8 = to_categorical(label_sf8 - tx_range[0])
    label_sf9 = to_categorical(label_sf9 - tx_range[0])

    data_sf7, label_sf7 = shuffle(data_sf7, label_sf7)
    data_sf8, label_sf8 = shuffle(data_sf8, label_sf8)
    data_sf9, label_sf9 = shuffle(data_sf9, label_sf9)

    snr_range = np.arange(0, 40)

    data_sf7, data_sf7_valid, label_sf7, label_sf7_valid = train_test_split(data_sf7, label_sf7, test_size=0.1,
                                                                            shuffle=True)
    data_sf8, data_sf8_valid, label_sf8, label_sf8_valid = train_test_split(data_sf8, label_sf8, test_size=0.1,
                                                                            shuffle=True)
    data_sf9, data_sf9_valid, label_sf9, label_sf9_valid = train_test_split(data_sf9, label_sf9, test_size=0.1,
                                                                            shuffle=True)

    data_source = [data_sf7, data_sf8, data_sf9]
    label_source = [label_sf7, label_sf8, label_sf9]

    num_train_samples = len(data_sf7) + len(data_sf8) + len(data_sf9)

    del data_sf7, data_sf8, data_sf9

    data_valid_source = [data_sf7_valid, data_sf8_valid, data_sf9_valid]
    label_valid_source = [label_sf7_valid, label_sf8_valid, label_sf9_valid]

    num_valid_samples = len(data_sf7_valid) + \
        len(data_sf8_valid) + len(data_sf9_valid)

    del data_sf7_valid, data_sf8_valid, data_sf9_valid

    batch_size = 32

    if model_type == ('lstm' or 'gru' or 'transformer'):
        train_generator = data_generator(data_source, label_source, batch_size, snr_range,
                                         data_type='sequential')
        valid_generator = data_generator(data_valid_source, label_valid_source, batch_size, snr_range,
                                         data_type='sequential')
    elif model_type == 'fcn':
        train_generator = data_generator(data_source, label_source, batch_size, snr_range,
                                         data_type='spatial')
        valid_generator = data_generator(data_valid_source, label_valid_source, batch_size, snr_range,
                                         data_type='spatial')

    # In[]
    '''Define the neural network'''
    if model_type == 'transformer':
        model = rff_transformer(maximum_position_encoding=254,
                                embed_dim=64, num_heads=8, num_classes=len(tx_range))
    elif model_type == 'lstm':
        model = lstm_model(num_classes=len(tx_range), embed_dim=64)
    elif model_type == 'gru':
        model = gru_model(num_classes=len(tx_range), embed_dim=64)
    elif model_type == 'fcn':
        model = FCN_model(num_classes=len(tx_range))

    '''Training configurations'''

    early_stop = EarlyStopping('val_loss', min_delta=0, patience=10)
    reduce_lr = ReduceLROnPlateau(
        'val_loss', min_delta=0, factor=0.2, patience=5, verbose=1)
    callbacks = [early_stop, reduce_lr]

    # opt = RMSprop(learning_rate=1e-4)
    opt = Adam(learning_rate=1e-3)

    model.compile(loss=['categorical_crossentropy'], optimizer=opt)

    history = model.fit(train_generator,
                        steps_per_epoch=num_train_samples // batch_size,
                        validation_data=valid_generator,
                        validation_steps=num_valid_samples // batch_size,
                        epochs=500,
                        verbose=1,
                        callbacks=callbacks)

    tf.keras.models.save_model(model, 'transformer_offline.h5')

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    # np.save('history.npy',history.history)
    # history=np.load('my_history.npy',allow_pickle='TRUE').item()

    return model


def inference(data_path, tx_range, model_type, model_path, snr_awgn):

    model = tf.keras.models.load_model(model_path, compile=False,
                                       custom_objects={'TransformerEncoder': TransformerEncoder,
                                                       'PositionEmbedding': PositionEmbedding})

    fuse_num = 1  # specify how many packets used in multi-packet RFFI protocol
    pkt_range = np.arange(0, 500, dtype=int)

    [data, label] = load_single_file(data_path,
                                     'data',
                                     'label',
                                     tx_range,
                                     pkt_range=np.arange(0, 500, dtype=int))

    label = label - tx_range[0]

    data = awgn(data, [snr_awgn])

    data = ch_ind_spectrogram(data, win_len=64, crop_ratio=0)

    if model_type == ('transformer' or 'gru' or 'lstm'):
        data = data[:, :, :, 0]
        data = data.transpose(0, 2, 1)  # [samples, timesteps, features]
    elif model_type == 'fcn':
        data = data

    pred_prob = model.predict(data)

    """Fuse multiple packets"""
    if fuse_num > 1:

        pred_prob_fused = pred_prob

        num_tx = len(tx_range)
        num_pkt = len(pkt_range)

        for dev_ind in range(num_tx):

            pred_per_dev = pred_prob[dev_ind*num_pkt:(dev_ind + 1)*num_pkt]

            pred_fused_per_dev = np.empty(pred_per_dev.shape)

            pred_fused_per_dev[0:fuse_num] = pred_per_dev[0:fuse_num]

            for pkt_ind in range(fuse_num, num_pkt):

                pred_fused_per_dev[pkt_ind] = np.mean(
                    pred_per_dev[pkt_ind-fuse_num:pkt_ind], axis=0)

            pred_prob_fused[dev_ind *
                            num_pkt:(dev_ind + 1)*num_pkt] = pred_fused_per_dev

        pred_prob = pred_prob_fused

    pred_label = pred_prob.argmax(axis=-1)
    conf_mat = confusion_matrix(label, pred_label)
    acc = accuracy_score(label, pred_label)
    print('Overall accuracy = %.4f' % acc)

    return acc, conf_mat


if __name__ == '__main__':

    # run_for = 'train'
    run_for = 'inference'

    if run_for == 'train':
        data_path = ['./Train/sf_7_train.h5',
                     './Train/sf_8_train.h5',
                     './Train/sf_9_train.h5']
        tx_range = np.arange(30, 40, dtype=int)
        model_type = 'gru'  # 'lstm' or 'gru' or 'transformer' or 'fcn'

        train(data_path, tx_range, model_type)

    elif run_for == 'inference':
        data_path = './sf_7_test.h5'
        tx_range = np.arange(30, 40, dtype=int)
        model_path = './nn_models/fcn_online.h5'
        model_type = 'fcn'
        snr_awgn = 30  # dB

        acc, cm = inference(data_path, tx_range, model_type, model_path, snr_awgn)
