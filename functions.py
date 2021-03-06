import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import LeakyReLU
from skimage.transform import resize

def lorentzian (x, a, x0, gam):
    '''
    Generate lorentzian curve.

    Parameters
    ----------
    x : array
        Range of function.
    a : float
        Amplitude.
    x0 : int
        Position of maximum.
    gam : float
        Full width at half maximum.

    Returns
    -------
    array
        Lorentzian curve to generate_signal.

    '''
    return a * gam**2 / ( gam**2 + ( x - x0 )**2)

def generate_signal(peaks):
    '''
    Generate one signal with n-peaks

    Parameters
    ----------
    peaks : int
        Number of peaks.

    Returns
    -------
    signal : array
        Signal with n-peaks.

    '''
    signal = 0
    x = np.linspace(-500,500,1000)
    for peak in range(peaks):
        a = random.uniform(10,50)
        x0 = np.random.randint(-300,300,1)[0]
        gam = random.uniform(10,30)
        signal = signal + lorentzian(x,a,x0,gam)
    return signal

def generate_data(size):
    '''
    Generate set of data do train and and fit model

    Parameters
    ----------
    size : int
        Size of output array.

    Returns
    -------
    clean_data : array
        Generated set of clean data without noise.
    noi_data : array
        Generated set of noised data.

    '''
    clean_data = []
    noi_data = []
    for s in range(size):
        peaks = np.random.randint(1,12,1)[0]
        noi = random.uniform(0.1,2)
        clean_signal = generate_signal(peaks)
        noi_signal = clean_signal+np.random.randn(len(clean_signal))*noi-(noi/2)
        clean_data.append(clean_signal)
        noi_data.append(noi_signal)
    return clean_data,noi_data

def reshape_data(X_train_noi,X_test_noi,X_train,X_test):
    '''
    Normalizing and reshaping data to fitting

    Parameters
    ----------
    X_train_noi : array
        Train data with noise.
    X_test_noi : array
        Test data with noise.
    X_train : array
        Train data without noise.
    X_test : array
        Test data without noise.

    Returns
    -------
    X_train_noi : array
        Normalized and reshaped data with noise to train.
    X_test_noi : array
        Normalized and reshaped data with noise to test.
    X_train : array
        Normalized and reshaped data without noise to train.
    X_test : array
        Normalized and reshaped data without noise to test.

    '''
    X_train = tf.keras.utils.normalize(X_train)
    X_test = tf.keras.utils.normalize(X_test)
    X_train_noi = tf.keras.utils.normalize(X_train_noi)
    X_test_noi = tf.keras.utils.normalize(X_test_noi)
    X_train = resize(np.array(X_train),(np.shape(X_train)[0],np.shape(X_train)[1],1))
    X_test = resize(np.array(X_test),(np.shape(X_test)[0],np.shape(X_test)[1],1))
    X_train_noi = resize(np.array(X_train_noi),(np.shape(X_train_noi)[0],np.shape(X_train_noi)[1],1))
    X_test_noi = resize(np.array(X_test_noi),(np.shape(X_test_noi)[0],np.shape(X_test_noi)[1],1))
    return X_train_noi,X_test_noi,X_train,X_test

def compile_model():
    '''
    Compile created architecture.

    Returns
    -------
    model : Tensorflow model
        Model to fit.

    '''
    inp = tf.keras.Input(shape = (1000,1))
    alpha = 0.015
    
    x = layers.Conv1D(64,90,strides = 1,padding ='same',activation= LeakyReLU(alpha = alpha))(inp)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(32,40,strides = 1,padding='same',activation=LeakyReLU(alpha = alpha))(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(16,30,1,'same',activation=LeakyReLU(alpha = alpha))(x)
    x = layers.MaxPool1D(2)(x)
    
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1DTranspose(16,30,1,'same',activation=LeakyReLU(alpha = alpha))(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1DTranspose(32,40,1,'same',activation=LeakyReLU(alpha = alpha))(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1DTranspose(64,90,1,'same',activation=LeakyReLU(alpha = alpha))(x)
    x = layers.Conv1DTranspose(1,15,padding ='same',activation='linear')(x)
    
    opt = tf.keras.optimizers.Adam(learning_rate = 0.000001)
    
    model = tf.keras.Model(inp,x)
    model.summary()
    model.compile(optimizer = opt,loss = 'mse')
    return model

def fit_model(model, X_train_noi,X_test_noi,X_train,X_test,batch_size = 4, epochs = 20):
    '''
    Fitting model to data.

    Parameters
    ----------
    model : Tensorflow model
        Model from compile_model.
    X_train_noi : array
        Train data with noise.
    X_test_noi : array
        Test data with noise.
    X_train : array
        Train data without noise.
    X_test : array
        Test data without noise.
    batch_size : int, optional
        Size of batch. The default is 2.
    epochs : int, optional
        Size of epochs. The default is 14.

    Returns
    -------
    model : Tensorflow model
        Fitted model, ready to predict.

    '''
    model.fit(
        x = X_train_noi,
        y= X_train,
        batch_size = batch_size,
        epochs = epochs,
        shuffle = True,
        validation_data = (X_test_noi,X_test)
    )
    return model

def save_model(model, path):
    '''
    Saving model from fit_model

    Parameters
    ----------
    model : Tensorflow model
        Model to save.
    path : str
        Path to dir.


    '''
    model.save(path)
    
def load_model(path):
    '''
    Load model from dir

    Parameters
    ----------
    path : str
        Path to model.

    '''
    return tf.keras.models.load_model(path)