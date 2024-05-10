import tensorflow as tf
import keras.backend as K
from keras.engine.topology import Layer
from keras.metrics import binary_accuracy
from keras.layers import Input, Dense, Concatenate, BatchNormalization, Dropout, Lambda
from keras.models import Model
from keras import regularizers
import pandas as pd
import numpy as np

def binary_classification_loss(concat_true, concat_pred):
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    t_pred = (t_pred + 0.001) / 1.002
    losst = tf.reduce_sum(K.binary_crossentropy(t_true, t_pred))

    return losst


def regression_loss(concat_true, concat_pred):
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]

    loss0 = tf.reduce_sum((1. - t_true) * tf.square(y_true - y0_pred))
    loss1 = tf.reduce_sum(t_true * tf.square(y_true - y1_pred))

    return loss0 + loss1


def ned_loss(concat_true, concat_pred):
    t_true = concat_true[:, 1]

    t_pred = concat_pred[:, 1]
    return tf.reduce_sum(K.binary_crossentropy(t_true, t_pred))


def dead_loss(concat_true, concat_pred):
    return regression_loss(concat_true, concat_pred)


def dragonnet_loss_binarycross(concat_true, concat_pred):
    return regression_loss(concat_true, concat_pred) + binary_classification_loss(concat_true, concat_pred)


def treatment_accuracy(concat_true, concat_pred):
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    return binary_accuracy(t_true, t_pred)

def track_epsilon(concat_true, concat_pred):
    epsilons = concat_pred[:, 3]
    return tf.abs(tf.reduce_mean(epsilons))


class EpsilonLayer(Layer):

    def __init__(self):
        super(EpsilonLayer, self).__init__()

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.epsilon = self.add_weight(name='epsilon',
                                       shape=[1, 1],
                                       initializer='RandomNormal',
                                       #  initializer='ones',
                                       trainable=True)
        super(EpsilonLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        # import ipdb; ipdb.set_trace()
        return self.epsilon * tf.ones_like(inputs)[:, 0:1]


def make_tarreg_loss(ratio=1., dragonnet_loss=dragonnet_loss_binarycross):
    def tarreg_ATE_unbounded_domain_loss(concat_true, concat_pred):
        vanilla_loss = dragonnet_loss(concat_true, concat_pred)

        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]

        y0_pred = concat_pred[:, 0]
        y1_pred = concat_pred[:, 1]
        t_pred = concat_pred[:, 2]

        epsilons = concat_pred[:, 3]
        t_pred = (t_pred + 0.01) / 1.02
        # t_pred = tf.clip_by_value(t_pred,0.01, 0.99,name='t_pred')

        y_pred = t_true * y1_pred + (1 - t_true) * y0_pred

        h = t_true / t_pred - (1 - t_true) / (1 - t_pred)

        y_pert = y_pred + epsilons * h
        targeted_regularization = tf.reduce_sum(tf.square(y_true - y_pert))

        # final
        loss = vanilla_loss + ratio * targeted_regularization
        return loss

    return tarreg_ATE_unbounded_domain_loss


def pairwise_distance(x, y, metric='euclidean'):

  distance_lookup = {
      'euclidean': lambda x, y: tf.linalg.norm(x - y, axis=1),
      'manhattan': lambda x, y: tf.reduce_sum(tf.abs(tf.math.subtract(x, y)), axis=1),
      'chebyshev': lambda x, y: tf.reduce_max(tf.abs(tf.math.subtract(x, y)), axis=1),
  }

  distance_fn = distance_lookup.get(metric, None)  # Handle potential metric not found

  if distance_fn is None:
      raise ValueError(f"Invalid metric: {metric}. Supported metrics: {', '.join(distance_lookup.keys())}")
  distances = distance_fn(x, y)
  return distances

def pairwise_distance_numpy(x, y, metric='euclidean'):

  distance_lookup = {
      'euclidean': lambda x, y: np.linalg.norm(x[:, None] - y, axis=2),
      'manhattan': lambda x, y: np.sum(np.abs(x[:, None] - y), axis=2),
      'chebyshev': lambda x, y: np.max(np.abs(x[:, None] - y), axis=2),
  }

  distance_fn = distance_lookup.get(metric, None)  # Handle potential metric not found

  if distance_fn is None:
      raise ValueError(f"Invalid metric: {metric}. Supported metrics: {', '.join(distance_lookup.keys())}")

  # Ensure at least one dimension for broadcasting
  if len(x.shape) == 1:
    x = x[:, None]
  if len(y.shape) == 1:
    y = y[:, None]

  distances = distance_fn(x, y)
  return distances



def make_dragonnet(input_dim, reg_l2,act_fn='elu'):
    """
    Dragonnet: https://github.com/claudiashi57/dragonnet
    """
    inputs = Input(shape=(input_dim,), name='input')

    # representation
    x = Dense(units=200, activation=act_fn, kernel_initializer='RandomNormal')(inputs)
    x = Dense(units=200, activation=act_fn, kernel_initializer='RandomNormal')(x)
    x = Dense(units=200, activation=act_fn, kernel_initializer='RandomNormal')(x)


    t_predictions = Dense(units=1, activation='sigmoid')(x)

    # HYPOTHESIS
    y0_hidden = Dense(units=100, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(x)
    y1_hidden = Dense(units=100, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(x)

    # second layer
    y0_hidden = Dense(units=100, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(y0_hidden)
    y1_hidden = Dense(units=100, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(y1_hidden)

    # third
    y0_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y0_predictions')(
        y0_hidden)
    y1_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y1_predictions')(
        y1_hidden)

    dl = EpsilonLayer()
    epsilons = dl(t_predictions, name='epsilon')

    concat_pred = Concatenate(1)([y0_predictions, y1_predictions, t_predictions, epsilons])
    model = Model(inputs=inputs, outputs=concat_pred)

    return model

def make_dragonnet_input_knn(input_dim, reg_l2,act_fn='elu'):
    """
    Dragonnet: https://github.com/claudiashi57/dragonnet
    """
    inputs = Input(shape=(input_dim,), name='input')

    # representation
    x = Dense(units=200, activation=act_fn, kernel_initializer='RandomNormal')(inputs)
    x = Dense(units=200, activation=act_fn, kernel_initializer='RandomNormal')(x)
    x = Dense(units=200, activation=act_fn, kernel_initializer='RandomNormal')(x)


    t_predictions = Dense(units=1, activation='sigmoid')(x)

    y_mean_0 = Input(shape=(1,),name='y_mean_0')
    y_mean_1 = Input(shape=(1,),name='y_mean_1')

    # HYPOTHESIS
    concat_y0 = Concatenate(1)([x, y_mean_0])
    concat_y1 = Concatenate(1)([x, y_mean_1])
    y0_hidden = Dense(units=100, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(concat_y0)
    y1_hidden = Dense(units=100, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(concat_y1)

    # second layer
    y0_hidden = Dense(units=100, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(y0_hidden)
    y1_hidden = Dense(units=100, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(y1_hidden)

    # third
    y0_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y0_predictions')(
        y0_hidden)
    y1_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y1_predictions')(
        y1_hidden)

    dl = EpsilonLayer()
    epsilons = dl(t_predictions, name='epsilon')
    # logging.info(epsilons)
    concat_pred = Concatenate(1)([y0_predictions, y1_predictions, t_predictions, epsilons])
    model = Model(inputs=[inputs,y_mean_0,y_mean_1], outputs=concat_pred)

    return model

def make_CBT(input_dim,
             reg_l2=0.01,
             act_fn='relu'):
    inputs = Input(shape=(input_dim,), name='input')

    # representation
    x = Dense(units=200, activation=act_fn, kernel_initializer='RandomNormal')(inputs)
    x = Dense(units=200, activation=act_fn, kernel_initializer='RandomNormal')(x)
    x = Dense(units=200, activation=act_fn, kernel_initializer='RandomNormal')(x)

    t_predictions = Dense(units=1, activation='sigmoid')(x)

    # HYPOTHESIS
    y0_hidden = Dense(units=100, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(x)
    y1_hidden = Dense(units=100, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(x)

    # second layer
    y0_hidden = Dense(units=100, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(y0_hidden)
    y1_hidden = Dense(units=100, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(y1_hidden)

    # third
    y0_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y0_predictions')(
        y0_hidden)
    y1_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y1_predictions')(
        y1_hidden)

    dl = EpsilonLayer()
    epsilons = dl(t_predictions, name='epsilon')
    # logging.info(epsilons)
    concat_pred = Concatenate(1)([y0_predictions, y1_predictions, t_predictions, epsilons])

    # Additional 'inputs' for the labels
    y_true = Input(shape=(1,), name='y_true')
    t_true = Input(shape=(1,), name='t_true')

    model = Model(inputs=[inputs, y_true, t_true], outputs=concat_pred)

    # L_CB loss
    t_pred = (t_predictions + 0.001) / 1.002

    loss0 = tf.reduce_sum((1. - t_true) * tf.square(y_true - y0_predictions))
    loss1 = tf.reduce_sum(t_true * tf.square(y_true - y1_predictions))

    regression_loss = loss0 + loss1

    vanilla_loss = regression_loss

    y_pred = t_true * y1_predictions + (1 - t_true) * y0_predictions

    h = t_true / t_pred - (1 - t_true) / (1 - t_pred)

    y_pert = y_pred + epsilons * h
    targeted_regularization = tf.reduce_sum(tf.square(y_true - y_pert))

    ones_to_sum = K.repeat_elements(t_true / t_pred, rep=input_dim, axis=1) * inputs
    zeros_to_sum = K.repeat_elements((1 - t_true) / (1 - t_pred), rep=input_dim, axis=1) * inputs

    ones_mean = tf.math.reduce_sum(ones_to_sum, 0) / tf.math.reduce_sum(t_true / t_pred, 0)
    zeros_mean = tf.math.reduce_sum(zeros_to_sum, 0) / tf.math.reduce_sum((1 - t_true) / (1 - t_pred), 0)

    loss = vanilla_loss + 0.5 * targeted_regularization + tf.keras.losses.mean_squared_error(zeros_mean,ones_mean)

    model.add_loss(loss)

    return model


SQRT_CONST = 1e-10
def safe_sqrt(x, lbound=SQRT_CONST):
    ''' Numerically safe version of TensorFlow sqrt '''
    return tf.sqrt(tf.clip_by_value(x, lbound, np.inf))

def pdist2sq(X,Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2*tf.matmul(X,tf.transpose(Y))
    nx = tf.reduce_sum(tf.square(X),1,keepdims=True)
    ny = tf.reduce_sum(tf.square(Y),1,keepdims=True)
    D = (C + tf.transpose(ny)) + nx
    return D

def pdist2(X,Y):
    """ Returns the tensorflow pairwise distance matrix """
    return safe_sqrt(pdist2sq(X,Y))

def lindisc(X,p,t):
    ''' Linear MMD '''

    it = tf.where(t>0)[:,0]
    ic = tf.where(t<1)[:,0]

    Xc = tf.gather(X,ic)
    Xt = tf.gather(X,it)

    mean_control = tf.reduce_mean(Xc,reduction_indices=0)
    mean_treated = tf.reduce_mean(Xt,reduction_indices=0)

    c = tf.square(2*p-1)*0.25
    f = tf.sign(p-0.5)

    mmd = tf.reduce_sum(tf.square(p*mean_treated - (1-p)*mean_control))
    mmd = f*(p-0.5) + safe_sqrt(c + mmd)

    return mmd

def mmd2_lin(X,t,p):
    ''' Linear MMD '''

    it = tf.where(t>0)[:,0]
    ic = tf.where(t<1)[:,0]

    Xc = tf.gather(X,ic)
    Xt = tf.gather(X,it)

    mean_control = tf.reduce_mean(Xc,reduction_indices=0)
    mean_treated = tf.reduce_mean(Xt,reduction_indices=0)

    mmd = tf.reduce_sum(tf.square(2.0*p*mean_treated - 2.0*(1.0-p)*mean_control))

    return mmd

def mmd2_rbf(X,t,p,sig):
    """ Computes the l2-RBF MMD for X given t """

    it = tf.where(t>0)[:,0]
    ic = tf.where(t<1)[:,0]

    Xc = tf.gather(X,ic)
    Xt = tf.gather(X,it)

    Kcc = tf.exp(-pdist2sq(Xc,Xc)/tf.square(sig))
    Kct = tf.exp(-pdist2sq(Xc,Xt)/tf.square(sig))
    Ktt = tf.exp(-pdist2sq(Xt,Xt)/tf.square(sig))

    m = tf.cast(tf.shape(Xc)[0], dtype=tf.float32)
    n = tf.cast(tf.shape(Xt)[0], dtype=tf.float32)

    mmd = tf.square(1.0-p)/(m*(m-1.0))*(tf.reduce_sum(Kcc)-m)
    mmd = mmd + tf.square(p)/(n*(n-1.0))*(tf.reduce_sum(Ktt)-n)
    mmd = mmd - 2.0*p*(1.0-p)/(m*n)*tf.reduce_sum(Kct)
    mmd = 4.0*mmd

    return mmd

def wasserstein(X,t,p, lam=10,its=10,sq=False,backpropT=False):
    """ Returns the Wasserstein distance between treatment groups """


    it = tf.where(t>0)[:,0]
    ic = tf.where(t<1)[:,0]
    Xc = tf.gather(X,ic)
    Xt = tf.gather(X,it)
    nc = tf.cast(tf.shape(Xc)[0], dtype=tf.float32)
    nt = tf.cast(tf.shape(Xt)[0], dtype=tf.float32)

    ''' Compute distance matrix'''
    if sq:
        M = pdist2sq(Xt,Xc)
    else:
        M = safe_sqrt(pdist2sq(Xt,Xc))


    ''' Estimate lambda and delta '''
    M_mean = tf.reduce_mean(M)
    #M_drop = tf.nn.dropout(M,10/(nc*nt))
    delta = tf.stop_gradient(tf.reduce_max(M))
    eff_lam = tf.stop_gradient(lam/M_mean)

    ''' Compute new distance matrix '''
    Mt = M
    row = delta*tf.ones(tf.shape(M[0:1,:]))
    #col = tf.concat(0,[delta*tf.ones(tf.shape(M[:,0:1])),tf.zeros((1,1))])
    delta_col = delta * tf.ones_like(M[:, 0:1])  # Create a column of ones with the same shape
    col = tf.concat([delta_col, tf.zeros((1, 1))], axis=0)
    Mt = tf.concat([M,row],0)
    Mt = tf.concat([Mt,col],1)

    ''' Compute marginal vectors '''

    a = tf.concat([p*tf.ones(tf.shape(tf.where(t>0)[:,0:1]))/nt, (1-p)*tf.ones((1,1))], 0)
    b = tf.concat([(1-p)*tf.ones(tf.shape(tf.where(t<1)[:,0:1]))/nc, p*tf.ones((1,1))], 0)

    ''' Compute kernel matrix'''
    Mlam = eff_lam*Mt
    K = tf.exp(-Mlam) + 1e-6 # added constant to avoid nan
    U = K*Mt
    ainvK = K/a

    u = a
    for i in range(0,its):
        u = 1.0/(tf.matmul(ainvK,(b/tf.transpose(tf.matmul(tf.transpose(u),K)))))
    v = b/(tf.transpose(tf.matmul(tf.transpose(u),K)))

    T = u*(tf.transpose(v)*K)

    if not backpropT:
        T = tf.stop_gradient(T)

    E = T*Mt
    D = 2*tf.reduce_sum(E)

    return D

def wasserstein1(Xt,Xc,t, p, lam=10,its=10,sq=False,backpropT=False):
    """ Returns the Wasserstein distance between treatment groups """

    nc = tf.cast(tf.shape(Xc)[0], dtype=tf.float32)
    nt = tf.cast(tf.shape(Xt)[0], dtype=tf.float32)

    ''' Compute distance matrix'''
    if sq:
        M = pdist2sq(Xt,Xc)
    else:
        M = safe_sqrt(pdist2sq(Xt,Xc))


    ''' Estimate lambda and delta '''
    M_mean = tf.reduce_mean(M)
    #M_drop = tf.nn.dropout(M,10/(nc*nt))
    delta = tf.stop_gradient(tf.reduce_max(M))
    eff_lam = tf.stop_gradient(lam/M_mean)

    ''' Compute new distance matrix '''
    Mt = M
    row = delta*tf.ones(tf.shape(M[0:1,:]))
    #col = tf.concat(0,[delta*tf.ones(tf.shape(M[:,0:1])),tf.zeros((1,1))])
    delta_col = delta * tf.ones_like(M[:, 0:1])  # Create a column of ones with the same shape
    col = tf.concat([delta_col, tf.zeros((1, 1))], axis=0)
    Mt = tf.concat([M,row],0)
    Mt = tf.concat([Mt,col],1)

    ''' Compute marginal vectors '''
    a = tf.concat([p*tf.ones(tf.shape(tf.where(t>0)[:,0:1]))/nt, (1-p)*tf.ones((1,1))], 0)
    b = tf.concat([(1-p)*tf.ones(tf.shape(tf.where(t<1)[:,0:1]))/nc, p*tf.ones((1,1))], 0)

    ''' Compute kernel matrix'''
    Mlam = eff_lam*Mt
    K = tf.exp(-Mlam) + 1e-6 # added constant to avoid nan
    U = K*Mt
    ainvK = K/a

    u = a
    for i in range(0,its):
        u = 1.0/(tf.matmul(ainvK,(b/tf.transpose(tf.matmul(tf.transpose(u),K)))))
    v = b/(tf.transpose(tf.matmul(tf.transpose(u),K)))

    T = u*(tf.transpose(v)*K)

    if not backpropT:
        T = tf.stop_gradient(T)

    E = T*Mt
    D = 2*tf.reduce_sum(E)

    return D

class CustomDragonnet(Model):
  def __init__(self, input_dim, act_fn, reg_l2):
    super(CustomDragonnet, self).__init__()

    # Shared representation layers
    self.representation_1 = Dense(units=200, activation=act_fn, kernel_initializer='RandomNormal')
    self.representation_2 = Dense(units=200, activation=act_fn, kernel_initializer='RandomNormal')
    self.representation_3 = Dense(units=200, activation=act_fn, kernel_initializer='RandomNormal')

    # Treatment prediction head
    self.treatment_prediction = Dense(units=1, activation='sigmoid')
    # Outcome prediction heads (one for each treatment)
    self.y0_hidden_1 = Dense(units=100, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))
    self.y0_hidden_2 = Dense(units=100, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))
    #self.y0_hidden_3 = Dense(units=100, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))
    self.y0_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y0_predictions')

    self.y1_hidden_1 = Dense(units=100, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))
    self.y1_hidden_2 = Dense(units=100, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))
    #self.y1_hidden_3 = Dense(units=100, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))
    self.y1_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y1_predictions')

    # Epsilon layer (remains unchanged)
    self.epsilon_layer = EpsilonLayer()

  def call(self, inputs):

    x = self.representation_1(inputs)
    x = self.representation_2(x)
    x = self.representation_3(x)


    y0_hidden = self.y0_hidden_1(x)
    y0_hidden = self.y0_hidden_2(y0_hidden)
    y0_predictions = self.y0_predictions(y0_hidden)

    y1_hidden = self.y1_hidden_1(x)
    y1_hidden = self.y1_hidden_2(y1_hidden)
    y1_predictions = self.y1_predictions(y1_hidden)

    # Epsilon layer (remains unchanged)
    t_predictions = self.treatment_prediction(x)
    epsilons = self.epsilon_layer(t_predictions)
    # Concatenate outputs (placeholder for incorporating KNN information)
    concat_pred = Concatenate(1)([y0_predictions, y1_predictions, t_predictions, epsilons])
    return concat_pred

def make_wasserstein(input_dim,
                reg_l2=0.01,
                act_fn='elu'):

    inputs = Input(shape=(input_dim,), name='input')

    # representation
    x = Dense(units=200, activation=act_fn, kernel_initializer='RandomNormal', name='x_representations1')(inputs)
    x = Dense(units=200, activation=act_fn, kernel_initializer='RandomNormal', name='x_representations2')(x)
    x = Dense(units=200, activation=act_fn, kernel_initializer='RandomNormal', name='x_representations3')(x)

    t_predictions = Dense(units=1, activation='sigmoid', name='sigmoid')(x)

    # HYPOTHESIS
    y0_hidden = Dense(units=100, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(x)
    y1_hidden = Dense(units=100, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(x)

    # second layer
    y0_hidden = Dense(units=100, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(y0_hidden)
    y1_hidden = Dense(units=100, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(y1_hidden)

    # third
    y0_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y0_predictions')(
        y0_hidden)
    y1_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y1_predictions')(
        y1_hidden)

    dl = EpsilonLayer()
    epsilons = dl(t_predictions, name='epsilon')
    concat_pred = Concatenate(1)([y0_predictions, y1_predictions, t_predictions, epsilons])

    y_true = Input(shape=(1,), name='y_true')
    t_true = Input(shape=(1,), name='t_true')

    model = Model(inputs=[inputs, y_true, t_true], outputs=concat_pred)

    t_pred = (t_predictions + 0.001) / 1.002
    loss0 = tf.reduce_sum((1. - t_true) * tf.square(y_true - y0_predictions))
    loss1 = tf.reduce_sum(t_true * tf.square(y_true - y1_predictions))

    regression_loss = loss0 + loss1
    vanilla_loss = regression_loss

    dist = wasserstein(x, t_true, 0.5)

    binary_classification_loss = tf.reduce_sum(K.binary_crossentropy(t_true, t_pred))

    y_pred = t_true * y1_predictions + (1 - t_true) * y0_predictions

    h = t_true / t_pred - (1 - t_true) / (1 - t_pred)

    y_pert = y_pred + epsilons * h
    targeted_regularization = tf.reduce_sum(tf.square(y_true - y_pert))

    loss = vanilla_loss + 1.5 * dist  + 0.2*targeted_regularization

    model.add_loss(loss)

    return model