<h2 align='center'>Naïve Bees: <br>Image Loading and Processing<br>Predicting Species from Images</h2>
<p align='center'><img src='https://www.dadant.com/wp-content/uploads/2017/12/ABJ-Extra-December12-256x256.jpg'></p>

![Forks](https://img.shields.io/github/forks/shukkkur/CodeForces.svg)
![Stars](https://img.shields.io/github/stars/shukkkur/CodeForces.svg)
![Watchers](https://img.shields.io/github/watchers/shukkkur/CodeForces.svg)
![Last Commit](https://img.shields.io/github/last-commit/shukkkur/CodeForces.svg) 

<p>Can a machine identify a bee as a honey bee or a bumble bee?</p>

<h3>Part 1: Image Loadning & Processing </h3>

<p>Required Libraries</p>

```python
from pathlib import Path
from PIL import Image
from IPython.display import display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```

<h3>1. Test Data</h3>

```python
test_data = np.random.beta(1, 1, size=(100, 100, 3))

plt.imshow(test_data)
```

<p align='left'><img src='https://github.com/shukkkur/Predict-Images-from-Species/blob/429e372a28e7409884bcf8a9b3f9c2ebb969a069/datasets/beta.jpg'></p>

<h3>2. Opening images with PIL</h3>

<p>Pillow is a very flexible image loading and manipulation library. </p>

```python
# open the image
img = Image.open('datasets/bee_1.jpg')

# Get the image size
img_size = img.size

print("The image size is: {}".format(img_size))

#display image
img

>>> The image size is: (100, 100)
```
<img src='datasets/bee_1.jpg'>


<h3>3. Image manipulation with PIL</h3>

<p>Pillow has a number of common image manipulation tasks built into the library.</p>
<ul>
<li>resizing</li>
<li>cropping</li>
<li>rotating</li>
<li>flipping</li>
<li>converting to greyscale</li>

<break>

```python
img_cropped = img.crop((25, 25, 75, 75))
display(img_cropped)

img_rotated = img.rotate(45,expand=25)
display(img_rotated)

img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
display(img_flipped)
```
<p align='left'><img src='https://github.com/shukkkur/Predict-Images-from-Species/blob/425b589d3780c89e514b7a334a590ca8e98c28d9/datasets/manipulation.jpg'></p>

<h3>4. Images as arrays of data</h3>

<p>Most image formats have three color "channels": red, green, and blue (some images also have a fourth channel called "alpha" that controls transparency). The way this is represented as data is as a three-dimensional matrix.</p>

```python
img_data = np.array(img)

img_data_shape = img_data.shape

print("Our NumPy array has the shape: {}".format(img_data_shape))

plt.imshow(img_data)
plt.show()

# plot the red channel
plt.imshow(img_data[:,:,0], cmap=plt.cm.Reds_r)
plt.show()

# plot the green channel
plt.imshow(img_data[:,:,1], cmap=plt.cm.Greens_r)
plt.show()

# plot the blue channel
plt.imshow(img_data[:,:,2], cmap=plt.cm.Blues_r)
plt.show()

>>> Our NumPy array has the shape: (100, 100, 3)
```

<img src='datasets/bees.jpg'>
  
<h3>5. Explore the color channels </h3>
<p>Color channels can help provide more information about an image. This kind of information can be useful when building models or examining the differences between images.<br>Let's look at the kernel density estimate for each of the color channels on the same plot so that we can understand how they differ.</p>

```python
def plot_kde(channel, color):
    """ Plots a kernel density estimate for the given data.
        
        `channel` must be a 2d array
        `color` must be a color string, e.g. 'r', 'g', or 'b'
    """
    data = channel.flatten()
    return pd.Series(data).plot.density(c=color)

# create the list of channels
channels = ['r', 'g', 'b']
    
def plot_rgb(image_data):
    # use enumerate to loop over colors and indexes
    for ix, color in enumerate(channels):
        plot_kde(img_data[:, :, ix], color)
    
    plt.show()
    
plot_rgb(img_data)
```
<img src='datasets/kernel.jpg'>


<h3>6. Honey bees and bumble bees</h3>
<p>Now we'll look at two different images and some of the differences between them.</p>

```python
honey = Image.open('datasets/bee_12.jpg')
bumble = Image.open('datasets/bee_3.jpg')

display(honey)
display(bumble)

honey_data = np.array(honey)
bumble_data = np.array(bumble)

plot_rgb(honey_data)
plot_rgb(bumble_data)
```

<p align='left'><img src='datasets/bee_3.jpg'></p>
<p align='left'><img src='datasets/kernel_honey.jpg'></p>
<p align='left'><img src='datasets/bee_12.jpg'></p>
<p align='left'><img src='datasets/kernel_bumble.jpg'></p>

<h3>8. Simplify</h3>
<p>We know that the colors of the flowers may be distracting from separating honey bees from bumble bees, so let's convert these images to black-and-white, or "grayscale." Because we change the number of color "channels," the shape of our array changes with this change. <p3>

```python
# convert honey to grayscale
honey_bw = honey.convert("L")

# convert the image to a NumPy array
honey_bw_arr = np.array(honey_bw)

honey_bw_arr_shape = honey_bw_arr.shape
print("Our NumPy array has the shape: {}".format(honey_bw_arr_shape))

plt.imshow(honey_bw_arr, cmap=plt.cm.gray)
plt.show()

plot_kde(honey_bw_arr, 'k')

>>> Our NumPy array has the shape: (100, 100)
```
<img src='datasets/simplify.jpg'></p>

<h3>Part 2: Building & Predicting </h3>
