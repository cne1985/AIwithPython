import numpy as np
from sklearn import preprocessing

input_data = np.array ([[5.1, -2.9, 3.3],[-1.2, 7.8, -6.1],[3.9, 0.4, 2.1],[7.3, -9.9, -4.5]])
print(input_data)

# Binarise data - converts numerical values to boolean, all values above 2,1 will become 1 all else 0
data_binarised = preprocessing.Binarizer(threshold=2.1).transform(input_data)
print("\nBinarised data:\n", data_binarised)

# Print mean and standard deviation
print("\nBEFORE:")
print("Mean =", input_data.mean(axis=0))
print("Std deviation =", input_data.std(axis=0))

# Remove mean
data_scaled = preprocessing.scale(input_data)
print("\nAFTER:")
print("Mean =", data_scaled.mean(axis=0))
print("Std deviation =", data_scaled.std(axis=0))

# Min max scaling
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print("\nMin max scaled data:\n", data_scaled_minmax)

# Normalise data
data_normalised_l1 = preprocessing.normalize(input_data, norm="l1")
data_normalised_l2 = preprocessing.normalize(input_data, norm="l2")
print("\nL1 normalised data:\n", data_normalised_l1)
print("\nL2 normalised data:\n", data_normalised_l2)