from modelhublib.processor import ImageProcessorBase
from torch.autograd import Variable
import PIL
import SimpleITK
import numpy as np
import json
from skimage.transform import resize
import math
import torch


class ImageProcessor(ImageProcessorBase):

    # OPTIONAL: Use this method to preprocess images using the image objects
    #           they've been loaded into automatically.
    #           You can skip this and just perform the preprocessing after
    #           the input image has been convertet to a numpy array (see below).
    def _preprocessBeforeConversionToNumpy(self, image):
        image = SimpleITK.ReadImage(image)
        if isinstance(image, PIL.Image.Image):
            # TODO: implement preprocessing of PIL image objects
            place_holder = True # Cannot have empty if blocks, syntax error
        elif isinstance(image, SimpleITK.Image):
            #----START---Resample image to common resolution of 1x1x1----START-----#
            new_spacing = [1,1,1]
            
            #Set up SitK resampling image filter
            rif = SimpleITK.ResampleImageFilter()
            rif.SetOutputSpacing(new_spacing)
            rif.SetOutputDirection(image.GetDirection())
            
            #Get original image size and spacing
            orig_size = np.array(image.GetSize(), dtype = np.int)
            orig_spacing = np.array(image.GetSpacing())
            
            #Calculate new image size based on current size and desired spacing. 
            new_size = np.ceil(orig_size*(orig_spacing/new_spacing)).astype(np.int)
            new_size = [int(s) for s in new_size]
            
            #Set up SitK resampling parameters
            rif.SetSize(new_size)
            rif.SetOutputOrigin(image.GetOrigin())
            rif.SetOutputPixelType(image.GetPixelID())
            rif.SetInterpolator(SimpleITK.sitkLinear)
            
            #Resample image and generate numpy array from image
            image = rif.Execute(image)
            image = SimpleITK.GetArrayFromImage(image)
            
        return image

    def _preprocessAfterConversionToNumpy(self, npArr):
        #--START-----Resize and Pad image to a uniform 256x256x256 voxel cube with retained aspect ratio----START----
        #Generate isotropic array of zeros based on maximum dimension of image
        pad = np.zeros((3,1))
        pad[0,0] = max(npArr.shape) - npArr.shape[0]
        pad[1,0] = max(npArr.shape) - npArr.shape[1]
        pad[2,0] = max(npArr.shape) - npArr.shape[2]

        paddedImage = np.zeros((max(npArr.shape),max(npArr.shape),max(npArr.shape)))
        #Pad image
        paddedImage = np.pad(npArr, ((int(math.ceil(pad[0,0]/2)), int(math.floor(pad[0,0]/2))),(int(math.ceil(pad[1,0]/2)), int(math.floor(pad[1,0]/2))),(int(math.ceil(pad[2,0]/2)), int(math.floor(pad[2,0]/2)))), 'constant', constant_values=0)
        #Resize padded image to desired size 256
        size_new = 256
        npArr = resize(paddedImage, (size_new, size_new, size_new), preserve_range = True)
        
        #--END-----Resize and Pad image to a uniform 256x256x256 voxel cube with retained aspect ratio----END----
        
        #---START---Transform to Tensor ---START
        npArr = np.expand_dims(npArr, axis=0)
        npArr = np.expand_dims(npArr, axis=0)
        npArr = torch.from_numpy(npArr.copy()).float()
        npArr = Variable(npArr)
        return npArr


    def computeOutput(self, inferenceResults):
        #compute the softmax probabilty of the image containing no dental artifact (result[0]) or containing a dental artifact (result[1])
        inference_result = inferenceResults.data.numpy()[0]
        e_x = np.exp(inference_result-np.max(inference_result))
        outputs_softmax = e_x/e_x.sum(axis = 0)
        result = np.array(outputs_softmax)
        return result
