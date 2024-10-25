import numpy as np

data_directory = './test_code_1/3D_data_total_np1.npz'
data = np.load(data_directory,allow_pickle=True)

X_all = data['X_all']
Y_all = data['Y_all']

X_all = X_all / 255

X_train, X_test, y_train, y_test = train_test_split(X_all, Y_all, test_size=0.20, random_state=42)

print(X_train.shape)
print(X_test.shape)