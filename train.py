import functions as f
from sklearn.model_selection import train_test_split
### Generate data
clean,noi = f.generate_data(32000)
### Split, normalize and reshape data
X_train_noi,X_test_noi,X_train,X_test = train_test_split(noi,clean,test_size =0.3)
X_train_noi,X_test_noi,X_train,X_test = f.reshape_data(X_train_noi,X_test_noi,X_train,X_test)
### Compile and fit model
model = f.compile_model()
model = f.fit_model(model, X_train_noi, X_test_noi, X_train, X_test)
### Save model  
f.save_model(model, 'model_denoise')