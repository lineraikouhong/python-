from sklearn.datasets import load_digits
digits=load_digits()
print(digits.data.shape)
print(digits.target.shape)
print(digits.images.shape)
import matplotlib.pyplot as plt
plt.matshow(digits.images[0])
plt.show()
