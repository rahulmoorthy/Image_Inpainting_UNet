# Image Inpainting using Deep Learning

• Deep learning based image inpainting implemented using U-Net.

• Given dataset contained a single image, hence performed random cropping of 128*128 patches. Applied rigorous data augmentation techniques to the input images.

• Designed a data-loader that yields a corrupted image and its mask as an input to the model.

• Trained U-Net (~100 epochs) model using MSE as loss function and Adam optimizer. 

• The model outputs an image like the ground truth image by restoring the corrupted/missing patch.

## Results:

### Images from left to right (ground-truth label, train input, train output)

![Epoch_1](/images/epoch1.JPG)

![Epoch_10](/images/epoch10.JPG)

![Epoch_20](/images/epoch20.JPG)

![Epoch_50](/images/epoch50.JPG)

![Epoch_99](/images/epoch99.JPG)
