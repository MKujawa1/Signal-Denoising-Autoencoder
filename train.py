import functions as f
from sklearn.model_selection import train_test_split

clean,noi = f.generate_data(10000)

X_train_noi,X_test_noi,X_train,X_test = train_test_split(noi,clean,test_size =0.3)
X_train_noi,X_test_noi,X_train,X_test = f.reshape_data(X_train_noi,X_test_noi,X_train,X_test)

model = f.compile_model()
model = f.fit_model(model, X_train_noi, X_test_noi, X_train, X_test)
    
f.save_model(model, 'model_denoise')