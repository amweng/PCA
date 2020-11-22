import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_lfw_people
lfw_people = fetch_lfw_people(min_faces_per_person=70)
n_samples, h, w = lfw_people.images.shape
npix = h*w
fea = lfw_people.data

def plt_face(x):
    global h,w
    plt.imshow(x.reshape((h, w)), cmap=plt.cm.gray)
    plt.xticks([])

plt.figure(figsize=(10,20))
nplt = 4
for i in range(nplt):    
    plt.subplot(1,nplt,i+1)
    plt_face(fea[i])

plt.show()




def compute_principal_compnents(fea):
    # Compute the mean down axis 0
    # the mean of each pixel across all images.
    Xmean = np.mean(fea,0) 
    plt_face(Xmean)
    # Zero-center the data (subtract the mean)
    feaCentered = fea - Xmean
    #NumPy svd returns V in VTranspose form already, so to get V, we transpose the output of np.linalg.svd.
    U, S, Vtranspose = np.linalg.svd(feaCentered, full_matrices= True)
    V = Vtranspose.T
    return Xmean,feaCentered,V

def project_face_onto_PCS(k, face_index,feaCentered, V):
    # where k = the number of principal components
    # get the principal components V[:,:k]
    # and project the fourth face onto them.
    Z = feaCentered[face_index].dot(V[:,:k])
    return Z

def project_face_onto_orig_space(Z,V,Xmean):
    # Project the face back into original space
    OrigWithoutMean = Z.dot(V[:,:k].T)
    # add back the mean so the image makes sense
    OrigMeanAdded = OrigWithoutMean + Xmean
    # display the approximate image
    plt_face(OrigMeanAdded)
    image = OrigMeanAdded
    return image

def do_PCA(face_index,k):
    Xmean, feaCentered, V = compute_principal_compnents(fea)
    Z = project_face_onto_PCS(k,face_index,feaCentered,V)
    image = project_face_onto_orig_space(Z,V,Xmean)
    plt.figure(figsize=(10,20))
    plt_face(image)
    plt.show()


face_index = 3
k = 100

do_PCA(face_index,k)

