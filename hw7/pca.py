import sys
import os
import numpy as np
from skimage import io
img = np.zeros((415, 600 * 600 * 3))
for i in range(415):
    img[i] = io.imread(os.path.join(sys.argv[1], '%d.jpg' % i)).astype(np.float32).reshape(-1)
mean = img.mean(axis=0)
img -= mean
U, sigma, V = np.linalg.svd(img, full_matrices=False)
idx = int(sys.argv[2][:-4])
print(idx)
p = U[idx:idx+1, :5].dot(np.diag(sigma[:5])).dot(V[:5]) + mean
p = p.astype(np.uint8).reshape((600, 600, 3))
io.imsave(sys.argv[3], p)