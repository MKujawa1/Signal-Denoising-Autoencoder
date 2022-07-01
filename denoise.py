import functions as f
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
### Load model
model = f.load_model('model_denoise')
### Generate data, normalize and reshape
ref,test = f.generate_data(1) 
ref_norm = tf.keras.utils.normalize(ref)
signals_test = tf.keras.utils.normalize(test)
signals_test = np.reshape(signals_test,(1,1000,1))
### Get denoised signal
pred = model.predict(signals_test)
pred = pred[0]
### Denormalizing
max_test = np.argmax(test[0])
mean_max_test = np.mean(test[0][max_test-2:max_test+2])
pred = pred*(mean_max_test/np.amax(pred))
### Results
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(test[0],label = 'Noised data')
plt.plot(ref[0],label = 'Ref data')
plt.ylabel('Amplitude')
plt.xlabel('# of points')
plt.tight_layout()
plt.grid(True)
plt.legend()
plt.subplot(1,2,2)
plt.plot(test[0],label = 'Noised data')
plt.plot(pred,label = 'Denoised data')
plt.ylabel('Amplitude')
plt.xlabel('# of points')
plt.tight_layout()
plt.grid(True)
plt.legend()
plt.show()

