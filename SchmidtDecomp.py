import numpy as np

# function to generate basis of noise vectors.  
# Uses Schmidt decomposition of the array of reference images.  
# INPUT: imgRef are the reference images (3D array -- a 1D array of 2D images).  
# OUTPUT: refBasis are the generated basis images (3D array -- a 1D array of 2D images).
def genBasis(imgRef):    
    refBasis=np.zeros((len(imgRef),255, 256))
    for i in range(0,len(imgRef)):
        refBasis[i,:,:]=imgRef[i,:,:]/np.sqrt(np.sum(imgRef[i,:,:]*imgRef[i,:,:]))
        for j in range(0,i):            
            coeff = np.sum(refBasis[i,:,:]*refBasis[j,:,:])            
            refBasis[i,:,:]=refBasis[i,:,:]-coeff*refBasis[j,:,:]
        refBasis[i,:,:]=refBasis[i,:,:]/np.sqrt(np.sum(refBasis[i,:,:]*refBasis[i,:,:]))
    return refBasis       

# function to generate matched reference images for absorption images
# Projects the raw absorption image onto the basis of noise vectors
# INPUT: imgAbs are the raw absorption images (3D array -- a 1D array of 2D images).  
# INPUT: refBasis is the basis of noise vectors (3D array -- a 1D array of 2D images).  
# OUTPUT: modRef are the generated matched reference images (3D array -- a 1D array of 2D images).
def genReference(imgAbs, refBasis):        
    modRef = np.zeros((len(imgAbs),255,256))
    for i in range(len(imgAbs)):        
        for j in range(len(refBasis)):    
            modRef[i,:,:] = modRef[i,:,:] + np.sum(imgAbs[i,:,:]*refBasis[j,:,:]) *refBasis[j,:,:]                
    return modRef
    