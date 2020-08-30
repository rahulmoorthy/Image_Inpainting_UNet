# Image Inpainting using Deep Learning

• Deep learning based image inpainting implemented using U-Net

• Given dataset contained a single image, hence performed random cropping of 128*128 patches. Applied rigorous data augmentation techniques to the input images.

• Designed a data-loader that yields a corrupted image and its mask as an input to the model.

• Trained U-Net (~100 epochs) model using MSE as loss function and Adam optimizer. The model outputs an image like the ground truth image by restoring the corrupted/missing patch.


![alt text](https://github.com/rahulmoorthy/Image_Inpainting_UNet/images/epoch1.jpg)
