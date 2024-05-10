
from models import *
import os, time
import glob
import argparse
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
import keras.backend as K
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, TerminateOnNaN
import pandas as pd
import numpy as np

from sklearn.metrics import pairwise_distances

def knn_by_treatment(x_train, t_train, y_train, k, metric='euclidean'):

    unique_treatments = np.unique(t_train)

    y0_means = np.full(x_train.shape[0], 0)
    y1_means = np.full(x_train.shape[0], 0)
    distances = pairwise_distances(x_train, metric=metric)

    for sample_index in range(x_train.shape[0]):
        treatment_0_indices = np.where(t_train == 0)[0]
        if len(treatment_0_indices) > 0:
            neighbor_indices = np.argsort(distances[sample_index, treatment_0_indices])[1:k + 1]
            neighbor_indices = treatment_0_indices[neighbor_indices]  
            neighbor_outcomes = y_train[neighbor_indices]
            y0_means[sample_index] = np.mean(neighbor_outcomes) 

        treatment_1_indices = np.where(t_train == 1)[0]
        if len(treatment_1_indices) > 0:
            neighbor_indices = np.argsort(distances[sample_index, treatment_1_indices])[1:k + 1]
            neighbor_indices = treatment_1_indices[neighbor_indices]  
            neighbor_outcomes = y_train[neighbor_indices]
            y1_means[sample_index] = np.mean(neighbor_outcomes) 

    return y0_means, y1_means


def _split_output_fredjo(yt_hat, t, y, y_scaler, x, mu_0, mu_1, split='TRAIN'):
    q_t0 = y_scaler.inverse_transform(yt_hat[:, 0].copy())
    q_t1 = y_scaler.inverse_transform(yt_hat[:, 1].copy())
    g = yt_hat[:, 2].copy()

    if yt_hat.shape[1] == 4:
        eps = yt_hat[:, 3][0]
    else:
        eps = np.zeros_like(yt_hat[:, 2])

    y = y_scaler.inverse_transform(y.copy())
    var = "{}: average propensity for treated: {} and untreated: {}".format(split,g[t.squeeze() == 1.].mean(),
                                                                        g[t.squeeze() == 0.].mean())
    print(var)

    return {'q_t0': q_t0, 'q_t1': q_t1, 'g': g, 't': t, 'y': y, 'x': x, 'mu_0': mu_0, 'mu_1': mu_1, 'eps': eps}

def train_and_predict_dragonnet(t_tr, y_tr, x_tr,mu_0_tr, mu_1_tr,
                          t_te, y_te, x_te,mu_0_te, mu_1_te,
                           output_dir='',
                            post_proc_fn=_split_output_fredjo):
    """
    https://github.com/claudiashi57/dragonnet
    """

    ###
    t_tr = t_tr.reshape(-1, 1)
    t_te = t_te.reshape(-1, 1)
    y_tr = y_tr.reshape(-1, 1)
    y_te = y_te.reshape(-1, 1)

    ###
    y_unscaled = np.concatenate([y_tr,y_te],axis=0)
    y_scaler = StandardScaler().fit(y_unscaled)
    y_tr = y_scaler.transform(y_tr)
    y_te = y_scaler.transform(y_te)
    train_outputs = []
    test_outputs = []

    print(">> I am DRAGONNET...")
    optim = 'adam'
    bs_ratio = 1.0
    val_split = 0.22
    batch_size=64
    verbose = 0
    ratio= 1.0
    act_fn = 'elu'
    dragonnet = make_dragonnet(x_tr.shape[1], 0.01,act_fn=act_fn)

    metrics = [regression_loss, binary_classification_loss, treatment_accuracy, track_epsilon]

    loss = make_tarreg_loss(ratio=ratio, dragonnet_loss=dragonnet_loss_binarycross)


    i = 0
    tf.random.set_seed(i)
    np.random.seed(i)

    x_train, x_test = x_tr, x_te
    y_train, y_test = y_tr, y_te
    t_train, t_test = t_tr, t_te

    yt_train = np.concatenate([y_train, t_train], 1)


    start_time = time.time()

    dragonnet.compile(
       optimizer=Adam(lr=1e-3),
       loss=loss, metrics=metrics)

    adam_callbacks = [
       TerminateOnNaN(),
       EarlyStopping(monitor='val_loss', patience=2, min_delta=0.),
       ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=verbose, mode='auto',
                         min_delta=1e-8, cooldown=0, min_lr=0)

    ]

    dragonnet.fit(x_train, yt_train, callbacks=adam_callbacks,
                 validation_split=val_split,
                 epochs=100,
                 batch_size=batch_size, verbose=verbose)

    sgd_callbacks = [
       TerminateOnNaN(),
       EarlyStopping(monitor='val_loss', patience=40, min_delta=0.),
       ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=verbose, mode='auto',
                         min_delta=0., cooldown=0, min_lr=0)
    ]

    sgd_lr = 1e-5
    momentum = 0.9
    dragonnet.compile(optimizer=SGD(lr=sgd_lr, momentum=momentum, nesterov=True), loss=loss,
                     metrics=metrics)
    dragonnet.fit(x_train, yt_train, callbacks=sgd_callbacks,
                 validation_split=val_split,
                 epochs=300,
                 batch_size=batch_size, verbose=verbose)

    elapsed_time = time.time() - start_time
    print("***************************** elapsed_time is: ", elapsed_time)

    yt_hat_test = dragonnet.predict(x_test)
    yt_hat_train = dragonnet.predict(x_train)

    test_outputs += [_split_output_fredjo(yt_hat_test, t_test, y_test, y_scaler, x_test, mu_0_te, mu_1_te,split='TEST')]
    train_outputs += [_split_output_fredjo(yt_hat_train, t_train, y_train, y_scaler, x_train, mu_0_tr, mu_1_tr,split='TRAIN')]
    K.clear_session()

    return test_outputs, train_outputs


def train_and_predict_cer(t_tr, y_tr, x_tr,mu_0_tr, mu_1_tr,
                          t_te, y_te, x_te,mu_0_te, mu_1_te,
                           output_dir='',
                            post_proc_fn=_split_output_fredjo):
    t_tr = t_tr.reshape(-1, 1)
    t_te = t_te.reshape(-1, 1)
    y_tr = y_tr.reshape(-1, 1)
    y_te = y_te.reshape(-1, 1)

    ###
    y_unscaled = np.concatenate([y_tr,y_te],axis=0)
    y_scaler = StandardScaler().fit(y_unscaled)
    y_tr = y_scaler.transform(y_tr)
    y_te = y_scaler.transform(y_te)
    train_outputs = []
    test_outputs = []

    print(">> I am CBT...")
    dragonnet = make_CBT(x_tr.shape[1],
                            reg_l2=0.01,
                            act_fn='relu')
    optim = 'sgd'
    lr = 1e-5
    momentum = 0.9
    bs_ratio = 1.0
    max_batch = 300
    val_split=0.22
    i = 0
    tf.random.set_seed(i)
    np.random.seed(i)

    x_train, x_test = x_tr, x_te
    y_train, y_test = y_tr, y_te
    t_train, t_test = t_tr, t_te

    yt_train = np.concatenate([y_train, t_train], 1)

    start_time = time.time()

    if optim == 'adam':
        dragonnet.compile(
        optimizer=Adam(lr=lr))
        adam_callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor='val_loss', patience=40, min_delta=0., restore_best_weights=True),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=0, mode='auto',
                              min_delta=1e-8, cooldown=0, min_lr=0)

        ]
        dummy = np.zeros((x_train.shape[0],))
        dragonnet.fit([x_train, y_train, t_train], dummy, callbacks=adam_callbacks,
                      validation_split=val_split,
                      epochs=max_batch,
                      batch_size=int(x_train.shape[0]*bs_ratio),
                      verbose=0)

    elif optim == 'sgd':
        sgd_callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor='val_loss', patience=40, min_delta=0. , restore_best_weights=True),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=0, mode='auto',
                              min_delta=0., cooldown=0, min_lr=0)
        ]

        dragonnet.compile(optimizer=SGD(lr=lr, momentum=momentum, nesterov=True))
        dummy = np.zeros((x_train.shape[0],))
        history = dragonnet.fit([x_train, y_train, t_train], dummy, callbacks=sgd_callbacks,
                      validation_split=val_split,
                      epochs=max_batch, #300
                      batch_size=int(x_train.shape[0]*bs_ratio),
                      verbose=0)

    else:
        raise Exception("optim <"+str(optim)+"> not supported!")

    elapsed_time = time.time() - start_time
    print("***************************** elapsed_time is: ", elapsed_time)

    dummy = np.zeros((x_test.shape[0],))
    yt_hat_test = dragonnet.predict([x_test,dummy,dummy])
    dummy = np.zeros((x_train.shape[0],))
    yt_hat_train = dragonnet.predict([x_train,dummy,dummy])

    test_outputs += [post_proc_fn(yt_hat_test, t_test, y_test, y_scaler, x_test, mu_0_te, mu_1_te,split='TEST')]
    train_outputs += [post_proc_fn(yt_hat_train, t_train, y_train, y_scaler, x_train, mu_0_tr, mu_1_tr,split='TRAIN')]
    K.clear_session()

    return test_outputs, train_outputs

def perform_psm(treatment, ps1, n_neighbors=1):
    it = tf.where(treatment>0)[:,0]
    ic = tf.where(treatment<1)[:,0]

    treated_ps = tf.gather(ps1, it)
    control_ps = tf.gather(ps1, ic)

    num_treated = tf.shape(treated_ps)[0]
    num_control = tf.shape(control_ps)[0]
    condition = tf.logical_and(tf.greater(num_treated, 0), tf.greater(num_control, 0))
    condition_met = tf.reduce_any(condition).numpy()
    if not condition_met:
        return None, None

    abs_dist = tf.abs(tf.expand_dims(treated_ps, 1) - tf.expand_dims(control_ps, 0))
    min_dist_indices = tf.argmin(abs_dist, axis=1)
    matched_control = tf.gather(ic, min_dist_indices)

    return it, matched_control
def train_and_predict_PSMWASS(t_tr, y_tr, x_tr,mu_0_tr, mu_1_tr,
                                t_te, y_te, x_te,mu_0_te, mu_1_te,
                                output_dir='',
                                post_proc_fn=_split_output_fredjo):
    t_tr = t_tr.reshape(-1, 1)
    t_te = t_te.reshape(-1, 1)
    y_tr = y_tr.reshape(-1, 1)
    y_te = y_te.reshape(-1, 1)

    ###
    y_unscaled = np.concatenate([y_tr,y_te],axis=0)
    y_scaler = StandardScaler().fit(y_unscaled)
    y_tr = y_scaler.transform(y_tr)
    y_te = y_scaler.transform(y_te)
    train_outputs = []
    test_outputs = []

    #tf.keras.backend.set_floatx('float32')
    print(">> I am PSMWASS...")
    dragonnet = CustomDragonnet(x_tr.shape[1], reg_l2=0.01, act_fn='elu')

    # for reporducing the experimemt
    i = 0
    tf.random.set_seed(i)
    np.random.seed(i)

    x_train, x_test = x_tr, x_te
    y_train, y_test = y_tr, y_te
    t_train, t_test = t_tr, t_te

    yt_train = np.concatenate([y_train, t_train], 1)

    start_time = time.time()

    if optim == 'adam':
        optimizer = Adam(learning_rate=lr)
    elif optim == 'sgd':
        optimizer = SGD(learning_rate=lr, momentum=momentum, nesterov=True)
    epochs = 300
    batch_size= 200
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train, t_train))
    train_dataset = train_dataset.shuffle(buffer_size=len(x_train)).batch(batch_size)

    for epoch in range(epochs):
        #print(f"\nEpoch: {epoch + 1}/{epochs}")
        # Training step
        for batch_data in train_dataset:
            with tf.GradientTape() as tape:
                inputs, y_true, t_true = batch_data
                inputs= tf.cast(inputs, dtype=tf.float32)
                y_true= tf.cast(y_true, dtype=tf.float32)
                t_true = tf.cast(t_true, dtype=tf.float32)
                predictions = dragonnet(inputs)
                y0_predictions = predictions[:,0]
                y1_predictions = predictions[:,1]
                t_pred = predictions[:,2]
                epsilons = predictions[:,3]

                loss0 = tf.reduce_sum((1. - t_true) * tf.square(y_true - y0_predictions))
                loss1 = tf.reduce_sum(t_true * tf.square(y_true - y1_predictions))

                regression_loss = loss0 + loss1
                binary_classification_loss = tf.reduce_sum(K.binary_crossentropy(t_true, t_pred))

                if use_bce:
                    vanilla_loss = regression_loss + binary_classification_loss
                else:
                    vanilla_loss = regression_loss

                y_pred = t_true * y1_predictions + (1 - t_true) * y0_predictions

                h = t_true / t_pred - (1 - t_true) / (1 - t_pred)

                y_pert = y_pred + epsilons * h
                targeted_regularization = tf.reduce_sum(tf.square(y_true - y_pert))

                matched_indx_tr, matched_indx_ctr = perform_psm(t_true, t_pred)
                batch_representation_1 = dragonnet.representation_1(inputs)
                batch_representation_2 = dragonnet.representation_2(batch_representation_1)
                batch_representation_3 = dragonnet.representation_3(batch_representation_2)

                if matched_indx_tr is None:
                    dist = wasserstein(batch_representation_3, t_true, 0.5)
                else:
                    treated = tf.gather(batch_representation_3, matched_indx_tr)
                    control = tf.gather(batch_representation_3, matched_indx_ctr)
                    ct = tf.concat([matched_indx_tr, matched_indx_ctr], axis=0)
                    treatment_extra = tf.gather(t_true, ct)
                    dist = wasserstein1(treated, control,treatment_extra, 0.5)

                ## final loss
                if use_targ_term:
                    loss = vanilla_loss + 0.5 * targeted_regularization + b_ratio * dist
                else:
                    loss = vanilla_loss + b_ratio * dist

                # Backpropagation and update
                gradients = tape.gradient(loss, dragonnet.trainable_variables)
                optimizer.apply_gradients(zip(gradients, dragonnet.trainable_variables))

    elapsed_time = time.time() - start_time
    print("***************************** elapsed_time is: ", elapsed_time)

    dummy = np.zeros((x_test.shape[0],))
    yt_hat_test = dragonnet.predict(x_test)
    dummy = np.zeros((x_train.shape[0],))
    yt_hat_train = dragonnet.predict(x_train)

    test_outputs += [post_proc_fn(yt_hat_test, t_test, y_test, y_scaler, x_test, mu_0_te, mu_1_te,split='TEST')]
    train_outputs += [post_proc_fn(yt_hat_train, t_train, y_train, y_scaler, x_train, mu_0_tr, mu_1_tr,split='TRAIN')]
    K.clear_session()

    return test_outputs, train_outputs

def train_and_predict_dragonnet_knn(t_tr, y_tr, x_tr,mu_0_tr, mu_1_tr,
                          t_te, y_te, x_te,mu_0_te, mu_1_te,
                           output_dir='',
                            post_proc_fn=_split_output_fredjo):
    """
     https://github.com/claudiashi57/dragonnet
    """

    ###
    t_tr = t_tr.reshape(-1, 1)
    t_te = t_te.reshape(-1, 1)
    y_tr = y_tr.reshape(-1, 1)
    y_te = y_te.reshape(-1, 1)

    ###
    y_unscaled = np.concatenate([y_tr,y_te],axis=0)
    y_scaler = StandardScaler().fit(y_unscaled)
    y_tr = y_scaler.transform(y_tr)
    y_te = y_scaler.transform(y_te)
    train_outputs = []
    test_outputs = []

    print(">> I am DRAGONNET...")
    optim = 'adam'
    bs_ratio = 1.0
    val_split = 0.22
    batch_size=64
    verbose = 0
    ratio= 1.0
    act_fn = 'elu'
    dragonnet = make_dragonnet_input_knn(x_tr.shape[1], 0.01, act_fn=act_fn)

    metrics = [regression_loss, binary_classification_loss, treatment_accuracy, track_epsilon]

    loss = make_tarreg_loss(ratio=ratio, dragonnet_loss=dragonnet_loss_binarycross)


    # for reporducing the experimemts
    i = 0
    tf.random.set_seed(i)
    np.random.seed(i)

    x_train, x_test = x_tr, x_te
    y_train, y_test = y_tr, y_te
    t_train, t_test = t_tr, t_te

    yt_train = np.concatenate([y_train, t_train], 1)


    start_time = time.time()

    dragonnet.compile(
       optimizer=Adam(lr=1e-3),
       loss=loss, metrics=metrics)

    adam_callbacks = [
       TerminateOnNaN(),
       EarlyStopping(monitor='val_loss', patience=2, min_delta=0.),
       ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=verbose, mode='auto',
                         min_delta=1e-8, cooldown=0, min_lr=0)

    ]

    y0_mean, y1_mean = knn_by_treatment(x_train, t_train, y_train, k=10, metric='euclidean')
    dragonnet.fit([x_train, y0_mean, y1_mean], yt_train, callbacks=adam_callbacks,
                  validation_split=val_split,
                  epochs=100,
                  batch_size=batch_size, verbose=verbose)


    sgd_callbacks = [
       TerminateOnNaN(),
       EarlyStopping(monitor='val_loss', patience=40, min_delta=0.),
       ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=verbose, mode='auto',
                         min_delta=0., cooldown=0, min_lr=0)
    ]

    sgd_lr = 1e-5
    momentum = 0.9
    dragonnet.compile(optimizer=SGD(lr=sgd_lr, momentum=momentum, nesterov=True), loss=loss,
                     metrics=metrics)
    dragonnet.fit([x_train, y0_mean, y1_mean], yt_train, callbacks=sgd_callbacks,
                  validation_split=val_split,
                  epochs=300,
                  batch_size=batch_size, verbose=verbose)

    elapsed_time = time.time() - start_time
    print("***************************** elapsed_time is: ", elapsed_time)

    y0_mean, y1_mean = knn_by_treatment(x_train, t_train, y_train, k=10, metric='euclidean')
    y0_mean_te, y1_mean_te = knn_by_treatment(x_test, t_test, y_test, k=10, metric='euclidean')
    yt_hat_test = dragonnet.predict([x_test, y0_mean_te, y1_mean_te])
    yt_hat_train = dragonnet.predict([x_train, y0_mean, y1_mean])

    test_outputs += [_split_output_fredjo(yt_hat_test, t_test, y_test, y_scaler, x_test, mu_0_te, mu_1_te,split='TEST')]
    train_outputs += [_split_output_fredjo(yt_hat_train, t_train, y_train, y_scaler, x_train, mu_0_tr, mu_1_tr,split='TRAIN')]
    K.clear_session()

    return test_outputs, train_outputs

def train_and_predict_wasserstein(t_tr, y_tr, x_tr,mu_0_tr, mu_1_tr,
                                t_te, y_te, x_te,mu_0_te, mu_1_te,
                                output_dir='',
                                post_proc_fn=_split_output_fredjo):
    t_tr = t_tr.reshape(-1, 1)
    t_te = t_te.reshape(-1, 1)
    y_tr = y_tr.reshape(-1, 1)
    y_te = y_te.reshape(-1, 1)

    ###
    y_unscaled = np.concatenate([y_tr,y_te],axis=0)
    y_scaler = StandardScaler().fit(y_unscaled)
    y_tr = y_scaler.transform(y_tr)
    y_te = y_scaler.transform(y_te)
    train_outputs = []
    test_outputs = []

    print(">> I am WASS...")
    act_fn = 'elu'
    optim = 'adam'
    lr = 1e-5
    #momentum = 0.9
    bs_ratio = 1.0
    max_batch = 300
    val_split = 0.22
    verbose = 0
    dragonnet = make_wasserstein(x_tr.shape[1],
                                reg_l2=0.01,
                                act_fn=act_fn)


    # for reporducing the experimemt
    i = 0
    tf.random.set_seed(i)
    np.random.seed(i)

    x_train, x_test = x_tr, x_te
    y_train, y_test = y_tr, y_te
    t_train, t_test = t_tr, t_te

    yt_train = np.concatenate([y_train, t_train], 1)

    start_time = time.time()

    if optim == 'adam':
        dragonnet.compile(
        optimizer=Adam(lr=lr))
        #dragonnet.summary()
        #exit(1)
        adam_callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor='val_loss', patience=40, min_delta=0., restore_best_weights=True),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=verbose, mode='auto',
                              min_delta=1e-8, cooldown=0, min_lr=0)

        ]
        dummy = np.zeros((x_train.shape[0],))
        dragonnet.fit([x_train, y_train, t_train], dummy, callbacks=adam_callbacks,
                      validation_split=val_split,
                      epochs=max_batch,
                      batch_size=300,
                      verbose=verbose)

    elif optim == 'sgd':
        sgd_callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor='val_loss', patience=40, min_delta=0. , restore_best_weights=True),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=verbose, mode='auto',
                              min_delta=0., cooldown=0, min_lr=0)
        ]

        dragonnet.compile(optimizer=SGD(lr=lr, momentum=momentum, nesterov=True))
        dummy = np.zeros((x_train.shape[0],))

        history = dragonnet.fit([x_train, y_train, t_train], dummy, callbacks=sgd_callbacks,
                      validation_split=val_split,
                      epochs=max_batch, #300
                      batch_size=int(x_train.shape[0]*bs_ratio),
                      verbose=verbose)

    else:
        raise Exception("optim <"+str(optim)+"> not supported!")

    elapsed_time = time.time() - start_time
    print("***************************** elapsed_time is: ", elapsed_time)

    dummy = np.zeros((x_test.shape[0],))
    yt_hat_test = dragonnet.predict([x_test,dummy,dummy])
    dummy = np.zeros((x_train.shape[0],))
    yt_hat_train = dragonnet.predict([x_train,dummy,dummy])

    test_outputs += [post_proc_fn(yt_hat_test, t_test, y_test, y_scaler, x_test, mu_0_te, mu_1_te,split='TEST')]
    train_outputs += [post_proc_fn(yt_hat_train, t_train, y_train, y_scaler, x_train, mu_0_tr, mu_1_tr,split='TRAIN')]
    K.clear_session()

    return test_outputs, train_outputs

def run(data_base_dir,
        output_dir,
        dragon):

    train_cv = np.load(os.path.join(data_base_dir,'ihdp_npci_1-1000.train.npz'))
    test = np.load(os.path.join(data_base_dir,'ihdp_npci_1-1000.test.npz'))

    X_tr    = train_cv.f.x.copy()
    T_tr    = train_cv.f.t.copy()
    YF_tr   = train_cv.f.yf.copy()
    YCF_tr  = train_cv.f.ycf.copy()
    mu_0_tr = train_cv.f.mu0.copy()
    mu_1_tr = train_cv.f.mu1.copy()

    X_te    = test.f.x.copy()
    T_te    = test.f.t.copy()
    YF_te   = test.f.yf.copy()
    YCF_te  = test.f.ycf.copy()
    mu_0_te = test.f.mu0.copy()
    mu_1_te = test.f.mu1.copy()

    T = np.concatenate([T_tr,T_te],axis=0)
    YF = np.concatenate([YF_tr,YF_te],axis=0)
    YCF = np.concatenate([YCF_tr,YCF_te],axis=0)
    mu_0_all = np.concatenate([mu_0_tr,mu_0_te],axis=0)
    mu_1_all = np.concatenate([mu_1_tr,mu_1_te],axis=0)

    for idx in range(X_tr.shape[-1]):
        print("++++",idx,"/",X_tr.shape[-1])

        t, y, y_cf, mu_0, mu_1 = T[:,idx], YF[:,idx], YCF[:, idx], mu_0_all[:,idx], mu_1_all[:,idx]

        ##################################
        simulation_output_dir = os.path.join(output_dir, str(idx))
        os.makedirs(simulation_output_dir, exist_ok=True)

        ##################################
        np.savez_compressed(os.path.join(simulation_output_dir, "simulation_outputs.npz"),
                            t=t, y=y, y_cf=y_cf, mu_0=mu_0, mu_1=mu_1)

        ##################################
        t_tr, y_tr, x_tr, mu0tr, mu1tr = T_tr[:,idx] , YF_tr[:,idx], X_tr[:,:,idx], mu_0_tr[:,idx], mu_1_tr[:,idx]
        t_te, y_te, x_te, mu0te, mu1te = T_te[:,idx] , YF_te[:,idx], X_te[:,:,idx], mu_0_te[:,idx], mu_1_te[:,idx]

        if dragon == 'CBT':
            test_outputs, train_output = train_and_predict_cer(t_tr, y_tr, x_tr, mu0tr, mu1tr,
                                                                    t_te, y_te, x_te, mu0te, mu1te,
                                                                    output_dir=simulation_output_dir)
            train_output_dir = os.path.join(simulation_output_dir, "CBT_RES")
            os.makedirs(train_output_dir, exist_ok=True)

            for num, output in enumerate(test_outputs):
                np.savez_compressed(os.path.join(train_output_dir, "{}_replication_test.npz".format(num)),
                                    **output)

            for num, output in enumerate(train_output):
                np.savez_compressed(os.path.join(train_output_dir, "{}_replication_train.npz".format(num)),
                                    **output)

        elif dragon == 'dragonnet':
            test_outputs, train_output = train_and_predict_dragonnet(t_tr, y_tr, x_tr, mu0tr, mu1tr,
                                                                      t_te, y_te, x_te, mu0te, mu1te,
                                                                        output_dir=simulation_output_dir)
            train_output_dir = os.path.join(simulation_output_dir, "DRAG_RES")
            os.makedirs(train_output_dir, exist_ok=True)

                # save the outputs of for each split (1 per npz file)
            for num, output in enumerate(test_outputs):
                    np.savez_compressed(os.path.join(train_output_dir, "{}_replication_test.npz".format(num)),
                                        **output)

            for num, output in enumerate(train_output):
                    np.savez_compressed(os.path.join(train_output_dir, "{}_replication_train.npz".format(num)),
                                        **output)
        elif dragon == 'WASS':
            test_outputs, train_output = train_and_predict_wasserstein(t_tr, y_tr, x_tr, mu0tr, mu1tr,
                                                                    t_te, y_te, x_te, mu0te, mu1te,
                                                                    output_dir=simulation_output_dir)
            train_output_dir = os.path.join(simulation_output_dir, "WASS_RES")
            os.makedirs(train_output_dir, exist_ok=True)

                # save the outputs of for each split (1 per npz file)
            for num, output in enumerate(test_outputs):
                np.savez_compressed(os.path.join(train_output_dir, "{}_replication_test.npz".format(num)),
                                        **output)

            for num, output in enumerate(train_output):
                np.savez_compressed(os.path.join(train_output_dir, "{}_replication_train.npz".format(num)),
                                        **output)
        elif dragon == 'KNN':
            test_outputs, train_output = train_and_predict_dragonnet_knn(t_tr, y_tr, x_tr, mu0tr, mu1tr,
                                                                       t_te, y_te, x_te, mu0te, mu1te,
                                                                       output_dir=simulation_output_dir)
            train_output_dir = os.path.join(simulation_output_dir, "KNN_RES")
            os.makedirs(train_output_dir, exist_ok=True)

            # save the outputs of for each split (1 per npz file)
            for num, output in enumerate(test_outputs):
                np.savez_compressed(os.path.join(train_output_dir, "{}_replication_test.npz".format(num)),
                                    **output)

            for num, output in enumerate(train_output):
                np.savez_compressed(os.path.join(train_output_dir, "{}_replication_train.npz".format(num)),
                                    **output)
        else:
            raise Exception("dragon:: "+str(dragon)+' not supported!!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_base_dir', type=str, help="path to data dir")
    parser.add_argument('--model', type=str, default='CBT', help="model")
    parser.add_argument('--output_base_dir', type=str, help="directory to save the output")
    args = parser.parse_args()

    output_dir = os.path.join(args.output_base_dir, args.model)
    run(data_base_dir=args.data_base_dir,
        output_dir=output_dir,
        dragon=args.model)

if __name__ == '__main__':
    main()