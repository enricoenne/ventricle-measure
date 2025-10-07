import numpy as np
import pandas as pd

from scipy.ndimage import distance_transform_edt, center_of_mass, label
from aicsimageio import AICSImage
import matplotlib.pyplot as plt

from skimage.draw import line

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

def get_valid(arr, height, width):
    valid = (arr[:,1] >= 0) & (arr[:,1] < height) & \
        (arr[:,0] >= 0) & (arr[:,0] < width)
    
    return arr[valid]

def get_ray_pixels(matrix, center, angle, res):

    max_length = np.linalg.norm(matrix.shape)

    r = np.arange(0, max_length)
    # all points along the ray
    x = ventr_coords[0] + r * np.cos(a)
    y = ventr_coords[1] + r * np.sin(a)
    pixels = np.vstack((y.astype(int), x.astype(int))).T

    pixels = get_valid(pixels, *matrix.shape)

    return(pixels)



def get_circle_pixels(matrix, center, radius, res):
    rows, cols = matrix.shape
    y, x = np.ogrid[:rows, :cols]
    cx, cy = center
    
    
    dist = np.sqrt(((y - cy)*res[1])**2 + ((x - cx)*res[0])**2)

    mask = np.isclose(dist, radius, atol=1)  # tolerance because exact match is rare

    return np.argwhere(mask)

def get_line_pixels(p1, p2):
    x0, y0 = p1
    x1, y1 = p2
    rr, cc = line(y0, x0, y1, x1)  # note: line() expects (row, col)
    return rr, cc

donut, res = get_picture2D('donut.tif')
point, _ = get_picture('point.tif')
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

# max value in the distance pic
max_dist = np.max(distance)


# find center of the ventricle
ventr_coords = center_of_mass(ventr_mask)


# find point in the reference pic
point_mask = point != 0
point_coords = center_of_mass(point_mask)


dx = point_coords[0] - ventr_coords[0]
dy = point_coords[1] - ventr_coords[1]
angle_0 = math.atan2(dy, dx)

max_length = np.linalg.norm(donut.shape)
# how many angles
n_angles = 40

results = pd.DataFrame(columns = ['angle_r', 'angle_d', 'thickness', 'ext_x', 'ext_y', 'vent_x', 'vent_y'])

test_img = distance.copy()

for i in range(n_angles):

    # angle of the ray
    a = 2*math.pi / n_angles * i + angle_0
    
    line_pixels = get_ray_pixels(donut, point_coords, a, res)

    cols = line_pixels[:,0]
    rows = line_pixels[:,1]



    # add radial picture line on top of picture
    # test_img[rows, cols] = max_dist

    # find the max ventricle distance along line
    current_thickness = np.max(distance[rows, cols])
    # where is the max distance along line
    current_index = np.argmax(distance[rows, cols])
    current_coords = [int(cols[current_index]), int(rows[current_index])]

    # circe centered in current point
    circ_pixels = get_circle_pixels(distance, current_coords, current_thickness, res)
    rows = circ_pixels[:,0]
    cols = circ_pixels[:,1]

    # add circle on top of picture
    # test_img[rows, cols] = max_dist

    # I need to set the pixels outside the tissue at some max value
    fake_distance = distance.copy()
    fake_distance[outside_mask] = max_dist
    vent_index = np.argmin(fake_distance[rows, cols])
    vent_coords = [int(cols[vent_index]), int(rows[vent_index])]

    r_line, c_line = get_line_pixels(current_coords, vent_coords)
    test_img[r_line, c_line] = max_dist

    results.loc[len(results)] = [a-angle_0, np.degrees(a-angle_0), current_thickness, *current_coords, *vent_coords]

# print(results)

plt.imshow(test_img, cmap = 'magma')
plt.scatter(results['ext_x'], results['ext_y'], s = 10, color = 'cyan')
plt.scatter(results['vent_x'], results['vent_y'], s = 10, color = 'cyan')
plt.show()

results.to_csv('result.csv')