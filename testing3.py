import numpy as np

# Load the array from the file
loaded_arr = np.load('my_array.npy')

# Print the shape of the array
array = loaded_arr.transpose(2, 1, 0).reshape(100, 18)

print("Shape of the numpy array:", array.shape)
np.savetxt('array_data.txt', array)