from modelhublib.processor import ImageProcessorBase
import PIL
import SimpleITK
import numpy as np
import json


class ImageProcessor(ImageProcessorBase):

    # OPTIONAL: Use this method to preprocess images using the image objects
    #           they've been loaded into automatically.
    #           You can skip this and just perform the preprocessing after
    #           the input image has been convertet to a numpy array (see below).
    def _preprocessBeforeConversionToNumpy(self, image):
        if isinstance(image, PIL.Image.Image):
            # TODO: implement preprocessing of PIL image objects
        elif isinstance(image, SimpleITK.Image):
            # TODO: implement preprocessing of SimpleITK image objects
        else:
            raise IOError("Image Type not supported for preprocessing.")
        return image


    def _preprocessAfterConversionToNumpy(self, npArr):
        # TODO: implement preprocessing of image after it was converted to a numpy array
        return npArr


    def computeOutput(self, inferenceResults):
        #compute the softmax probabilty of the image containing no dental artifact (result[0]) or containing a dental artifact (result[1])
        inference_result = inferenceResults.data.numpy()[0]
        e_x = np.exp(inference_result-np.max(inference_result))
        outputs_softmax = e_x/e_x.sum(axis = 0)
        result = np.array(outputs_softmax)
        return result
