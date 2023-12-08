import numpy as np
no_clusters=50
image_count=7
im_features = np.array([np.zeros(no_clusters) for i in range(image_count)])
print(im_features.shape)