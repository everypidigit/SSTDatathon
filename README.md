# Fire Detection with a CNN

## A repository that contains all of the files for the SST case from the Google Datathon


The sst_fire256.ipynb notebook contains the code for the neural network itself. It is based on the Cifar10CNN architecture, a really simple one. It was written and executed in Google Colab, so some folderpaths are weird. Should be no problem to run it on a local machine, though. Just change paths. The images were also resized to a 256x256 pixel images. It let me run the code on Google Colab since its RAM did not let me use larger images, and my laptop could not handle even 256x256 ones.

The sorter.ipynb notebook contains the code to sort the training images folder. Since they were not sorted beforehand, I wanted to sort them in separate folders: 0 for 'no fire' and 1 for 'with fire'.

Some of the files and folders, namely labels.csv and the training and test ones, were renamed because it was easier for me to use them in that way. Training images folder was renamed to "train", testing to "test", and the labels file was renamed to "labels.csv". No files were added or deleted.  

All of the images were resized to 256x256 pixels size, including the test images. 

Average accuracy on the validation set is 0.90703125 after 10 epochs of training.
