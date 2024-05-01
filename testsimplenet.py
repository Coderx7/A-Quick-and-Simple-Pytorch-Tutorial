#%%
# simplenet_cifar_5m_extra_pool
# import sys
# print(sys.version)
# import torch
# from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
# from simplenet import SimpleNet, simplenet_cifar_5m_extra_pool, simplenet_cifar_5m
# from simplenet2 import simplenet
# print(f'{torch.__version__}')


# model0 = simplenet()
# model = simplenet_cifar_5m_extra_pool(num_classes=10)

# print(f'{model}')
# print(f'{model0}')
# state_dict = torch.load("/media/hossein/SSD1/code_dl/best_chkpt_simplenet_cifar10_2018-12-25_00-34-57.pth.tar", map_location=lambda storage, loc: storage)['state_dict']
# # Create a new state dictionary without the "module." prefix
# state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
# print('after purge:')
# # remove the dorpout layers otherthan the ones after maxpools 
# def remove_dropout_layers(model):
#     features = torch.nn.Sequential()
#     prev_layer = None
#     i=0
#     for layer in model.features.children():
#         if isinstance(layer, torch.nn.Dropout2d):
#             if not isinstance(prev_layer, torch.nn.MaxPool2d):
#                 continue
#             layer = torch.nn.Dropout2d(0.1)
#         prev_layer = layer
#         features.add_module(str(i), layer);i+=1
#     # assign the updated features
#     model.features = features
#     return model

# model = remove_dropout_layers(model)
# # load the new state_dict into the model
# model.load_state_dict(state_dict)

import unittest
# for creating key press for waitKey
from unittest.mock import patch
import numpy as np
import cv2

class TestOpenCVGUI(unittest.TestCase):
    
    def test_imshow(self):
        # Create a simple black image
        image = 255 * np.ones((100, 100, 3), dtype=np.uint8)
        # Display the image in a window
        cv2.imshow('Test Window', image)
        # Check if the window was created
        self.assertEqual(cv2.getWindowProperty('Test Window', cv2.WND_PROP_VISIBLE), 1)
        # Close the window
        cv2.destroyAllWindows()

    # we use mock to simulate a key press so the test can go on
    # without any user interaction
    @patch('cv2.waitKey', return_value=ord('q'))
    def test_waitKey(self, mock_waitKey):
        # Call the cv2.waitKey() 
        key = cv2.waitKey(0)
        # Check if the key returned is the same as the mock
        self.assertEqual(key, ord('q'))

    def test_namedWindow(self):
        # Create a named window
        cv2.namedWindow('TestWindow')
        # Check if the window was created
        self.assertEqual(cv2.getWindowProperty('TestWindow', cv2.WND_PROP_VISIBLE), 1)
        # Destroy the window
        cv2.destroyWindow('TestWindow')

    def test_sift_keypoints(self):
        # Create a checkerboard pattern
        image = np.kron([[1, 0] * 4, [0, 1] * 4] * 4, np.ones((50, 50))).astype(np.uint8) * 255
        # for versions>4.4 for lower versions we use cv2.xfeatures2d.SIFT_create )
        sift = cv2.SIFT_create()
        # Detect keypoints
        keypoints = sift.detect(image, None)
        # Check if keypoints are detected (there should be 330 keypoints!)
        self.assertTrue(len(keypoints) == 330)

    def test_daisy_descriptor(self):
        # Create a checkerboard pattern
        image = np.kron([[1, 0] * 4, [0, 1] * 4] * 4, np.ones((50, 50))).astype(np.uint8) * 255
        # Initialize DAISY descriptor
        daisy = cv2.xfeatures2d.DAISY_create()
        # Compute descriptors
        keypoints = cv2.KeyPoint_convert([[100, 100]])
        descriptors = daisy.compute(image, keypoints)
        # Check if descriptors were computed
        self.assertTrue(descriptors is not None)

if __name__ == '__main__':
    unittest.main()
