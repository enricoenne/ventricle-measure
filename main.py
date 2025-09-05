import numpy as np
import pandas as pd

from scipy.ndimage import distance_transform_edt, center_of_mass, label
from aicsimageio import AICSImage
import matplotlib.pyplot as plt

import math

def get_picture(path):
    '''
    reading picture and metadata

    output[0] picture
    output[1] spatial relotuion in x, y and z
    '''
    obj = AICSImage(path)
    #print(f'size (y, x, z): {img.GetSize()}')

    # axis order before t - c - z - y - x
    img = obj.data
    # axis order after x - y - z
    return (np.transpose(img[0, 0, :, :, :], axes=(1, 2, 0)), (obj.physical_pixel_sizes.X, obj.physical_pixel_sizes.Y, obj.physical_pixel_sizes.Z))

def get_picture2D(path):
    '''
    reading picture and metadata

    output[0] picture
    output[1] spatial relotuion in x, y and z
    '''
    obj = AICSImage(path)

    # axis order before t - c - z - y - x
    img = obj.data
    img = np.squeeze(img)
    # axis order after x - y - z
    return (img, (obj.physical_pixel_sizes.X, obj.physical_pixel_sizes.Y))

def show_pic(pic, title = ''):
    plt.imshow(pic)
    plt.title(title)
    plt.show()


donut, res = get_picture2D('donut.tif')
height, width = donut.shape
print(width, height)

donut_mask = donut != 0

# we should have only ventricle and outside regions
labeled, n = label(~ donut_mask)

if n != 2:
    raise ValueError('problem with masks, too many regions')

# outside label is the label of the pixel on the top left
outside_label = labeled[(0,0)]
outside_mask = labeled == outside_label
# ventricle is the other region
ventr_mask = (labeled != outside_label) & (~ donut_mask)


# show_pic(donut_mask, 'donut')

# calculating distance from ventricle
distance = np.zeros_like(donut, dtype=np.float64)
# keeping it only in the area of the donut
distance[donut_mask] = distance_transform_edt(~ventr_mask, sampling= res)[donut_mask]

# show_pic(distance)

ventr_coords = center_of_mass(ventr_mask)


point, _ = get_picture('point.tif')
point_mask = point != 0
point_coords = center_of_mass(point_mask)


plt.imshow(distance)
plt.scatter(ventr_coords[1], ventr_coords[0], color='#FF8800')
plt.scatter(point_coords[1], point_coords[0], color='#008800')
plt.show()

dx = point_coords[0] - ventr_coords[0]
dy = point_coords[1] - ventr_coords[1]
angle_0 = math.atan2(dy, dx)

max_length = np.linalg.norm(donut.shape)
n_angles = 40

results = pd.DataFrame(columns = ['angle_r', 'angle_d', 'thickness'])

test_img = distance.copy()

for i in range(n_angles):
    a = 2*math.pi / n_angles * i + angle_0

    r = np.arange(0, max_length)

    x = ventr_coords[0] + r * np.cos(a)
    y = ventr_coords[1] + r * np.sin(a)

    pixels = np.vstack((y.astype(int), x.astype(int))).T

    valid = (pixels[:,1] >= 0) & (pixels[:,1] < height) & \
            (pixels[:,0] >= 0) & (pixels[:,0] < width)
    pixels = pixels[valid]


    cols = pixels[:,0]
    rows = pixels[:,1]

    test_img[rows, cols] = 3000

    current_thickness = np.max(distance[rows, cols])

    results.loc[len(results)] = [a-angle_0, np.degrees(a-angle_0), current_thickness]

# print(results)

# plt.imshow(test_img)
# plt.show()

results.to_csv('result.csv')