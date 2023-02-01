import os
import numpy as np
import time
import logging
from sklearn.metrics import roc_curve, accuracy_score, auc
import sklearn
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.ensemble import RandomForestClassifier
from sktime.transformations.panel.rocket import Rocket
import scipy.stats as ss
from sklearn.pipeline import make_pipeline
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

from utils.plot import plot_my_roc_version1, plot_loss_freq_small
from utils.utils import logging_best_model
from utils.logging_set import LoggingSet

import tensorflow as tf
keras = tf.keras  # import tensorflow.keras as keras


class Classifier_NN:
    def __init__(self, output_directory, config, logging, NN_name_str='ResNet'):
        self.output_directory = output_directory
        self.min_lr           = config.opt.min_lr
        self.patience         = config.opt.patience
        if NN_name_str.upper() == 'RESNET':
            self.model = self.build_model_resnet(config.model.input_layer_size, config.model.output_layer_size)
        elif NN_name_str.upper() == 'MLP':
            self.model = self.build_model_mlp(config.model.input_layer_size, config.model.output_layer_size)
        self.batch_size       = config.training.batch_size
        self.max_epochs       = config.training.max_epochs
        self.verbose          = False
        self.logging          = logging

    def build_model_resnet(self, input_shape, nb_classes):
        n_feature_maps = 64
        input_layer = keras.layers.Input(input_shape)

        # BLOCK 1
        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)

        # BLOCK 2
        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation('relu')(output_block_2)

        # BLOCK 3
        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = keras.layers.BatchNormalization()(output_block_2)

        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation('relu')(output_block_3)

        # FINAL
        gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=self.patience, min_lr=self.min_lr)

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss',
                                                           save_best_only=True)

        self.callbacks = [reduce_lr, model_checkpoint]

        return model

    def build_model_mlp(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)

        # flatten/reshape because when multivariate all should be on the same axis
        input_layer_flattened = keras.layers.Flatten()(input_layer)
        hidden_shape = input_shape
        hidden_shape = 500
        layer_1 = keras.layers.Dropout(0.1)(input_layer_flattened)
        layer_1 = keras.layers.Dense(hidden_shape, activation='relu')(layer_1)

        layer_2 = keras.layers.Dropout(0.2)(layer_1)
        layer_2 = keras.layers.Dense(hidden_shape, activation='relu')(layer_2)

        layer_3 = keras.layers.Dropout(0.2)(layer_2)
        layer_3 = keras.layers.Dense(hidden_shape, activation='relu')(layer_3)

        output_layer = keras.layers.Dropout(0.3)(layer_3)
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(output_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=self.patience, min_lr=self.min_lr)

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss',
                                                           save_best_only=True)

        self.callbacks = [reduce_lr, model_checkpoint]

        return model

    def fit(self, x_train, y_train, x_val, y_val):
        if len(tf.config.list_physical_devices('GPU'))==0:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! No GPU is found !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')


        mini_batch_size = int(min(x_train.shape[0] / 10, self.batch_size))
        start_time = time.time()
        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=self.max_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)
        duration = time.time() - start_time
        self.model.save(self.output_directory + 'last_model.hdf5')

        logging_best_model(self.logging, hist, duration)
        plot_loss_freq_small(self.output_directory,
                             history_val_loss_freq_small=hist.history['val_loss'],
                             history_train_loss_freq_small=hist.history['loss'],
                             history_lr=hist.history['lr'])

        keras.backend.clear_session()

    def predict(self, x_test):
        model_path = self.output_directory + 'best_model.hdf5'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test)
        return y_pred

    def delete_saved_model(self):
        model_path = self.output_directory + 'best_model.hdf5'
        os.remove(model_path)
        model_path = self.output_directory + 'last_model.hdf5'
        os.remove(model_path)


def NN_fit_predict(ParaImportant, *, para_dict_tuple, data_prefix_tuple,
                   label_true, LRT_hiddentrue, LRT_simpleapprox, name_prefix, feature_triple_np,
                   LRT_low_frequency=None, new_trainvaltest_tuple=None):
    """
       name_prefix:
        prefix of the file ='ResNet_' or 'MLP_'
    """
    print("@@@@@@@@  ----NN---- started: @@@@@@@@@@@@")
    (para_dict1, para_dict2)     = para_dict_tuple
    (data_prefix1, data_prefix2) = data_prefix_tuple

    logging_save_path_pre, logging_save_path = LoggingSet(logging, para_dict_tuple=(para_dict1, para_dict2),
                                                          data_prefix_tuple=(data_prefix1, data_prefix2),
                                                          loggingname_prestr=name_prefix)
    if 'num_dims' in para_dict1:
        num_dims = para_dict1['num_dims']
    else:
        num_dims = 1
    if len(feature_triple_np.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        feature_triple_np = feature_triple_np.reshape((feature_triple_np.shape[0], num_dims+1, para_dict1['num_steps'])).transpose((0, 2, 1))

    # Get config
    Mysize_dict  = {
        'input_layer_size': feature_triple_np.shape[1:],
        'output_layer_size': len(np.unique(label_true)),
        'num_all_samples': LRT_hiddentrue.shape[0]
    }
    configclass  = ParaImportant(Mysize_dict=Mysize_dict)
    config       = configclass.config

    if new_trainvaltest_tuple is None:
        all_ind_list    = list(range(0, config.training.num_all_samples, 1))
        train_ind_list  = np.random.choice(all_ind_list, size=config.training.num_tra_samples_withval, replace=False)
        val_ind_list    = np.random.choice(np.setdiff1d(all_ind_list, train_ind_list), size=config.training.num_val_samples_withval, replace=False)
        test_ind_list   = list(np.setdiff1d(np.setdiff1d(all_ind_list, train_ind_list), val_ind_list))
    else:
        train_size, val_size, test_size, train_size_withoutval = new_trainvaltest_tuple
        all_ind_list    = list(range(0, config.training.num_all_samples, 1))
        train_ind_list  = np.random.choice(all_ind_list, size=train_size, replace=False)
        val_ind_list    = np.random.choice(np.setdiff1d(all_ind_list, train_ind_list), size=val_size, replace=False)
        test_ind_list   = np.random.choice(np.setdiff1d(np.setdiff1d(all_ind_list, train_ind_list), val_ind_list), size=test_size, replace=False)

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(label_true.reshape(-1, 1))
    label_true_hot = enc.transform(label_true.reshape(-1, 1)).toarray()

    x_train = feature_triple_np[train_ind_list, :]
    y_train = label_true_hot[train_ind_list, :]
    x_test  = feature_triple_np[test_ind_list, :]
    # y_test  = label_true_hot[test_ind_list, :]
    y_test_true = label_true[test_ind_list]
    x_val   = feature_triple_np[val_ind_list, :]
    y_val   = label_true_hot[val_ind_list, :]
    # y_val_true  = label_true[val_ind_list]

    output_directory = logging_save_path[:-4] + '_'
    resnet_class = Classifier_NN(output_directory, config, logging, NN_name_str=name_prefix[:-1])
    resnet_class.fit(x_train, y_train, x_val, y_val)
    res = resnet_class.predict(x_test)
    resnet_class.delete_saved_model()

    fpr_test_NNpred, tpr_test_NNpred, thresholds_test_NNpred           = roc_curve(y_test_true, -res[:, 0]+res[:, 1], drop_intermediate=True)
    fpr_simpleapprox, tpr_simpleapprox, thresholds_simpleapprox     = roc_curve(label_true, -LRT_simpleapprox, drop_intermediate=True)
    fpr_hiddentrue, tpr_hiddentrue, thresholds_hiddentrue           = roc_curve(label_true, -LRT_hiddentrue, drop_intermediate=True)
    if LRT_low_frequency is not None:
        fpr_low_frequency, tpr_low_frequency, thresholds_low_frequency = roc_curve(label_true, -LRT_low_frequency, drop_intermediate=True)
    else:
        fpr_low_frequency = None
        tpr_low_frequency = None
        thresholds_low_frequency = None

    plot_my_roc_version1(logging_save_path, addition_str='test', fpr_NNpred=fpr_test_NNpred, tpr_NNpred=tpr_test_NNpred,
                                                fpr_simpleapprox=fpr_simpleapprox, tpr_simpleapprox=tpr_simpleapprox,
                                                fpr_low_frequency=fpr_low_frequency, tpr_low_frequency=tpr_low_frequency,
                                                fpr_hiddentrue=fpr_hiddentrue, tpr_hiddentrue=tpr_hiddentrue)

    return fpr_test_NNpred, tpr_test_NNpred



def RF_fit_predict(ParaImportant, *, para_dict_tuple, data_prefix_tuple,
                       label_true, LRT_hiddentrue, LRT_simpleapprox, LRT_low_frequency=None,
                       feature_triple_np=None, new_trainvaltest_tuple=None):

    print("@@@@@@@@ Test for ----RFforBinary---- started: @@@@@@@@@@@@")
    (para_dict1, para_dict2)     = para_dict_tuple
    (data_prefix1, data_prefix2) = data_prefix_tuple

    # Get config
    Mysize_dict  = {
        'input_layer_size': 0,  # useless for this case
        'output_layer_size': 0,  # useless for this case
        'num_all_samples': feature_triple_np.shape[0]
    }
    configclass  = ParaImportant(Mysize_dict=Mysize_dict)
    config       = configclass.config

    if new_trainvaltest_tuple is None:
        all_ind_list         = list(range(0, config.training.num_all_samples, 1))
        train_ind_list       = np.random.choice(all_ind_list, size=config.training.num_tra_samples, replace=False)
        test_ind_list        = list(np.setdiff1d(all_ind_list, train_ind_list))
    else:
        train_size, val_size, test_size, train_size_withoutval = new_trainvaltest_tuple
        all_ind_list         = list(range(0, config.training.num_all_samples, 1))
        train_ind_list       = np.random.choice(all_ind_list, size=train_size_withoutval, replace=False)
        test_ind_list        = np.random.choice(np.setdiff1d(all_ind_list, train_ind_list), size=test_size, replace=False)


    # Get train set and test set
    feature_triple_train_np = feature_triple_np[train_ind_list, :]
    label_true_train     = label_true[train_ind_list]
    feature_triple_test_np = feature_triple_np[test_ind_list, :]
    label_true_test       = label_true[test_ind_list]

    # Get logging
    logging_save_path_pre, logging_save_path = LoggingSet(logging, para_dict_tuple=(para_dict1, para_dict2), data_prefix_tuple=(data_prefix1, data_prefix2),
                                                          loggingname_prestr='RFforBinary')

    """
    Random Forest Classifier
    """
    logging.info("Random Forest start")
    RandomForest_Class = RandomForestClassifier()
    param_dist = {
        "n_estimators": ss.randint(10, 100),
        "max_depth": [3, None],
        "max_features": ss.randint(1, 11),
        "min_samples_split": ss.randint(2, 11),
        "bootstrap": [True, False],
        "criterion": ["gini", "entropy"],
    }
    rsh = HalvingRandomSearchCV(estimator=RandomForest_Class, param_distributions=param_dist, factor=2, cv=5)
    rsh.fit(feature_triple_train_np, label_true_train)
    best_para = rsh.best_params_
    logging.info(best_para)

    RandomForest_Class.set_params(**best_para)
    RandomForest_Class.fit(feature_triple_train_np, label_true_train)

    test_pred_binary_np_2dim = RandomForest_Class.predict_proba(feature_triple_test_np)
    test_pred_binary_np = test_pred_binary_np_2dim[:, 1]
    # train_pred_binary_np_2dim = RandomForest_Class.predict_proba(feature_triple_train_np)
    # train_pred_binary_np = train_pred_binary_np_2dim[:, 1]

    """
    Plot
    """
    # Plot of ROC
    fpr_test_NNpred, tpr_test_NNpred, thresholds_test_NNpred = roc_curve(label_true_test, test_pred_binary_np, drop_intermediate=True)
    fpr_simpleapprox, tpr_simpleapprox, thresholds_simpleapprox = roc_curve(label_true, -LRT_simpleapprox, drop_intermediate=True)
    if not (LRT_low_frequency is None):
        fpr_low_frequency, tpr_low_frequency, thresholds_low_frequency  = roc_curve(label_true, -LRT_low_frequency, drop_intermediate=True)
    else:
        fpr_low_frequency = None
        tpr_low_frequency = None
        thresholds_low_frequency = None
    fpr_hiddentrue, tpr_hiddentrue, thresholds_hiddentrue = roc_curve(label_true, -LRT_hiddentrue, drop_intermediate=True)
    plot_my_roc_version1(logging_save_path, addition_str='test', fpr_NNpred=fpr_test_NNpred, tpr_NNpred=tpr_test_NNpred,
                                                fpr_simpleapprox=fpr_simpleapprox, tpr_simpleapprox=tpr_simpleapprox,
                                                fpr_low_frequency=fpr_low_frequency, tpr_low_frequency=tpr_low_frequency,
                                                fpr_hiddentrue=fpr_hiddentrue, tpr_hiddentrue=tpr_hiddentrue)

    print("@@@@@@@@ Test for ----RFforBinary---- finished: @@@@@@@@@@@@")

    return fpr_test_NNpred, tpr_test_NNpred


def ROCKET_fit_predict(ParaImportant, *, para_dict_tuple, data_prefix_tuple,
                       label_true, LRT_hiddentrue, LRT_simpleapprox, LRT_low_frequency=None,
                       feature_triple_np=None, output1=None, new_trainvaltest_tuple=None):

    print("@@@@@@@@ Test for ----ROCKRTforBinary---- started: @@@@@@@@@@@@")
    (para_dict1, para_dict2)     = para_dict_tuple
    (data_prefix1, data_prefix2) = data_prefix_tuple

    # Get config
    Mysize_dict  = {
        'input_layer_size': 0,  # useless for this case
        'output_layer_size': 0,  # useless for this case
        'num_all_samples': feature_triple_np.shape[0]
    }
    configclass  = ParaImportant(Mysize_dict=Mysize_dict)
    config       = configclass.config


    # Get train set and test set. For ROCKET: the Input : 3D np.ndarray of shape = [n_instances, n_dimensions, series_length]
    if new_trainvaltest_tuple is None:
        all_ind_list         = list(range(0, config.training.num_all_samples, 1))
        train_ind_list       = np.random.choice(all_ind_list, size=config.training.num_tra_samples, replace=False)
        test_ind_list         = list(np.setdiff1d(all_ind_list, train_ind_list))
    else:
        train_size, val_size, test_size, train_size_withoutval = new_trainvaltest_tuple
        all_ind_list         = list(range(0, config.training.num_all_samples, 1))
        train_ind_list       = np.random.choice(all_ind_list, size=train_size_withoutval, replace=False)
        test_ind_list        = np.random.choice(np.setdiff1d(all_ind_list, train_ind_list), size=test_size, replace=False)

    if output1.shape.__len__() == 3:
        num_dims = output1.shape[2]
    else:
        num_dims = 1
    feature_triple_np_forROCKET       = feature_triple_np.reshape((feature_triple_np.shape[0], output1.shape[1], -1)).transpose(0, 2, 1)  # This is WRONG, but better!!
    # feature_triple_np_forROCKET       = feature_triple_np.reshape((feature_triple_np.shape[0], num_dims + 1, output1.shape[1]))  # This is TRUE, but perform bad
    feature_triple_train_np_forROCKET = feature_triple_np_forROCKET[train_ind_list, :, :]
    feature_triple_test_np_forROCKET   = feature_triple_np_forROCKET[test_ind_list, :, :]
    label_true_train        = label_true[train_ind_list]
    label_true_test          = label_true[test_ind_list]

    # Get logging
    logging_save_path_pre, logging_save_path = LoggingSet(logging, para_dict_tuple=(para_dict1, para_dict2), data_prefix_tuple=(data_prefix1, data_prefix2),
                                                          loggingname_prestr='ROCKETforBinary')

    """
    ROCKET and Ridge Classifier
    """
    logging.info("Rocket + Ridge: Rocket start")
    rocket = Rocket()
    rocket.fit(feature_triple_train_np_forROCKET, y=label_true_train)
    feature_triple_train_np_transform = rocket.transform(feature_triple_train_np_forROCKET)
    feature_triple_test_np_transform = rocket.transform(feature_triple_test_np_forROCKET)

    logging.info("Ridge start")

    RidgeClassifier_Class = make_pipeline(StandardScaler(with_mean=False), linear_model.RidgeCV(alphas=np.logspace(-3+4, 3+4, 20)))
    RidgeClassifier_Class.fit(feature_triple_train_np_transform, label_true_train)

    # RidgeClassifier_Class = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), cv=10)
    # standard_scaler = StandardScaler(with_mean=False)
    # standard_scaler.fit(feature_triple_train_np_transform)
    # feature_triple_train_np_transform = standard_scaler.transform(feature_triple_train_np_transform)
    # RidgeClassifier_Class.fit(feature_triple_train_np_transform, label_true_train)
    # logging.info(RidgeClassifier_Class)

    test_pred_binary_np = RidgeClassifier_Class.predict(feature_triple_test_np_transform)
    # RidgeClassifier_Class['ridgecv'].intercept_
    logging.info("alpha is %.2f" % RidgeClassifier_Class['ridgecv'].alpha_)
    # RidgeClassifier_Class["standardscaler"].transform(RidgeClassifier_Class["ridgecv"].coef_.reshape(1, -1))

    # train_pred_binary_np = RidgeClassifier_Class.predict(feature_triple_train_np_transform)

    """
    Plot
    """
    # Plot of ROC
    fpr_test_NNpred, tpr_test_NNpred, thresholds_test_NNpred = roc_curve(label_true_test, test_pred_binary_np, drop_intermediate=True)
    fpr_simpleapprox, tpr_simpleapprox, thresholds_simpleapprox = roc_curve(label_true, -LRT_simpleapprox, drop_intermediate=True)
    if not (LRT_low_frequency is None):
        fpr_low_frequency, tpr_low_frequency, thresholds_low_frequency  = roc_curve(label_true, -LRT_low_frequency, drop_intermediate=True)
    else:
        fpr_low_frequency = None
        tpr_low_frequency = None
        thresholds_low_frequency = None
    fpr_hiddentrue, tpr_hiddentrue, thresholds_hiddentrue = roc_curve(label_true, -LRT_hiddentrue, drop_intermediate=True)
    plot_my_roc_version1(logging_save_path, addition_str='test', fpr_NNpred=fpr_test_NNpred, tpr_NNpred=tpr_test_NNpred,
                                                fpr_simpleapprox=fpr_simpleapprox, tpr_simpleapprox=tpr_simpleapprox,
                                                fpr_low_frequency=fpr_low_frequency, tpr_low_frequency=tpr_low_frequency,
                                                fpr_hiddentrue=fpr_hiddentrue, tpr_hiddentrue=tpr_hiddentrue)

    print("@@@@@@@@ Test for ----ROCKRTforBinary---- finished: @@@@@@@@@@@@")

    return fpr_test_NNpred, tpr_test_NNpred