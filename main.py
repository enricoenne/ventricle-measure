import numpy as np
import pandas as pd

from scipy.ndimage import distance_transform_edt, center_of_mass, label
from scipy.signal import convolve2d
from aicsimageio import AICSImage
import matplotlib.pyplot as plt

from skimage.draw import line

import math

from scipy.spatial import ConvexHull
from matplotlib.path import Path

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
    x = center[0] + r * np.cos(angle)
    y = center[1] + r * np.sin(angle)
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


'''
this part is for weird shapes
'''
from scipy.ndimage import gaussian_filter

def smoothing(pic, smoothing_factor = None):
    area_pos = np.sum(pic == np.max(pic))

    if smoothing_factor is None:
        smoothing_factor = area_pos**(1/2) / 50
    print(f'smoothing factor = {smoothing_factor:.2f}')

    return gaussian_filter(pic, sigma=smoothing_factor, mode='constant', cval=0) > np.max(pic)/2

def find_edge(outside_m):
    kernel = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ])

    neighbor_sum = convolve2d(outside_m, kernel, mode='same', boundary='fill', fillvalue=0)

    edge = (outside_m == 0) & (neighbor_sum > 0)

    return edge

def follow_gradient(distance, start, mask_in, max_steps=1000):
    """
    Follow the distance gradient from start until reaching mask_in.
    start: tuple (row, col)
    mask_in: boolean array of internal edge
    """

    max_dist = 2 * np.max(distance)
    dist_data = np.ma.filled(distance, max_dist)


    gy, gx = np.gradient(dist_data)
    
    # it could have non valid gradient due to the mask
    pos = start.astype(float)

    pos_list = []


    for _ in range(max_steps):
        r, c = int(round(pos[0])), int(round(pos[1]))
        pos_list.append((r,c))
        # plt.scatter(c, r, s=0.1, c="#ffe600")
        if mask_in[r,c]:
            # plt.show()
            return (r, c), pos_list
        
        # Gradient at current position
        dr, dc = gy[tuple([r,c])], gx[tuple([r,c])]

        
        # Move in direction of steepest descent
        norm = np.hypot(dr, dc)
        if norm == 0:
            # plt.show()
            return (r, c), pos_list
        pos -= np.array([dr, dc]) / norm * 1  # small step can be scaled if needed

    # plt.show()
    return (r, c), pos_list


def df_col_to_points(col):
    coords = np.array(col)
  
    rows = coords[:, 0]
    cols = coords[:, 1]

    return cols, rows

def thickening(pic, th = 5):
    pic = pic.astype(bool)

    dist_pic = distance_transform_edt(~pic)

    return (dist_pic < th)

def find_angle(p,q):

    dx = p[0] - q[0]
    dy = p[1] - q[1]
    
    return math.atan2(dy, dx)

def ordered_edge_points(edge):
    coords_in_np = np.argwhere(edge)


    start_p = tuple(coords_in_np[0])
    ordered_p = [start_p]

    while len(ordered_p) < coords_in_np.shape[0]:

        current_p = ordered_p[-1]

        neighborhood = edge[current_p[0]-1:current_p[0]+2, current_p[1]-1:current_p[1]+2]
        
        # the set contains the considered point and the two 
        neighbors = [tuple(int(c) for c in k) for k in np.argwhere(neighborhood)]

        neighbors.remove((1,1))

        # these are the only possible configurations for the first pixel
        # ---    ---    ---
        # -xx    -xx    -x-
        # x--    -x-    x-x
        # the only possible anticlockwise pixels are (2,0) or (2,1)
        # 
        if len(ordered_p) == 1:
            if (2,0) in neighbors:
                next_p = (2,0)
            elif (2,1) in neighbors:
                next_p = (2,1)
            else:
                raise ValueError('something wrong with the edge')
            
            # we need to convert the point coordinates back to the absolute reference
            next_p = np.array([c-1 for c in next_p])
            next_p = tuple(current_p + next_p)
        else:
            for p in neighbors:
                # we need to convert the point coordinates back to the absolute reference
                p = np.array([c-1 for c in p])
                p = tuple(current_p + p)

                if p not in ordered_p:
                    next_p = p
                    break
            else:
                raise ValueError('something wrong with the visited pixels')

        ordered_p.append(next_p)
    
    return ordered_p

def plot_ordered(order_var, y_var, df, name=None):
    df_sorted = df.sort_values(by=order_var)

    plt.title(order_var)
    plt.plot(df_sorted[order_var], df_sorted[y_var])

    if 'angle_mark' in df.columns:
        angle_marks = df.loc[df_sorted.loc[df_sorted['angle_mark'] == 1].index, order_var]
    elif 'quadrant' in df.columns:
        angle_marks = df.loc[np.flatnonzero(np.diff(df_sorted['quadrant']) > 0), order_var]

    plt.vlines(angle_marks, ymin=0, ymax=df[y_var].max()*1.1, linestyles='--', color='#aaaaaa')

    if name is not None:
        plt.savefig(f'plots/{name}_{y_var}-{order_var}.png')
    plt.show()

def find_closest(p, edge, res=(1,1)):
    blank = np.zeros_like(edge)
    blank[p] = 1

    d = distance_transform_edt(~blank, sampling=res)
    d = np.ma.masked_array(d, ~edge)

    output_coords = tuple(int(c) for c in np.unravel_index(np.argmin(d), edge.shape))
    dist = float(d[output_coords])

    return output_coords, dist

def follow_gradient_quantification(distance, start, edge_in, edge_out, res = (1,1), max_steps=1000, show = False):
    """
    Follow the distance gradient from start until reaching mask_in.
    start: tuple (row, col)
    mask_in: boolean array of internal edge
    """

    flag_in = edge_in[start[0], start[1]]
    flag_out = edge_out[start[0], start[1]]

    if flag_in:
        distance = -distance
        target_edge = edge_out
    elif flag_out:
        target_edge = edge_in
    else:
        target_edge = edge_in
    

    max_dist = 2 * np.max(distance)
    dist_data = np.ma.filled(distance, max_dist)

    gy, gx = np.gradient(dist_data)
    

    # it could have non valid gradient due to the mask
    pos = start.astype(float)

    length = 0
    pos_list = []

    if show:
        plt.imshow(distance)
        plt.xlim(start[1]-200,start[1]+200)
        plt.ylim(start[0]-200,start[0]+200)
    
    prev_move = np.zeros(2)
    alpha = 0.6

    for _ in range(max_steps):
        r, c = int(round(pos[0])), int(round(pos[1]))
        pos_list.append((r,c))
        if show:
            plt.scatter(c, r, s=0.1, c="#ff00c8")
        # condition to return if we reach the opposite edge
        if target_edge[r,c] == 1:
            if show:
                plt.show()
            
            return length, (r, c), pos_list
        
        # gradient at current position
        dr, dc = gy[tuple([r,c])], gx[tuple([r,c])]

        
        # Move in direction of steepest descent
        norm = np.hypot(dr * res[0], dc * res[1])
        if norm == 0:
            # I DONT KNOW WHAT IS THE PROBLEM, WHY IS IT STOPPING

            # print((r,c))
            # sometimes it might get stuck in points that don't belong to the edge
            proper_coord, dist_from_target = find_closest((r, c), target_edge, res=res)

            # for some readon adding this distance causes a lot of problems
            # length += dist_from_target

            if show:
                plt.show()
            return length, proper_coord, pos_list
        
        movement = np.array([dr, dc]) / (norm + 1e-8)
        movement = alpha * prev_move + (1 - alpha) * movement
        pos -= movement
        prev_move = movement

        length += np.hypot(res[0], res[1])


    proper_coord, dist_from_target = find_closest((r, c), target_edge, res=res)
    if show:
        plt.show()
    return length, (r, c), pos_list


# GYRIFICATION

def get_edge_coords(shape):
    edge = ordered_edge_points(find_edge(shape))
    edge = np.array(edge)

    xs = edge[:,0]
    ys = edge[:,1]

    return xs, ys

def dist(p0, p1, res=(1,1)):
    return np.hypot((p1[0]-p0[0])*res[0],
                    (p1[1]-p0[1])*res[1])

def len_edge(shape, res=(1,1), loop=True):
    edge = find_edge(shape)

    edge = ordered_edge_points(edge)

    l = 0

    for i, p in enumerate(edge):
        if i==0:
            if loop:
                l_step = dist(p, edge[-1], res=res)
            else:
                l_step = 0
        else:
            l_step = dist(p, edge[i-1], res=res)

        l += l_step

    return float(l)

def gyr_index(donut, res=(1,1)):
    donut = np.pad(donut, pad_width=10, mode='constant', constant_values=0)

    donut_mask = donut != 0

    # we should have only ventricle and outside regions
    labeled, n = label(~ donut_mask)

    if n != 2:
        raise ValueError('problem with masks, too many regions')

    # outside label is the label of the pixel on the top left
    outside_label = labeled[(0,0)]
    outside_mask = labeled == outside_label

    tissue = ~outside_mask

    points = np.column_stack(np.nonzero(tissue))

    hull = ConvexHull(points)
    hull_coords = points[hull.vertices]

    yy, xx = np.mgrid[0:tissue.shape[0], 0:tissue.shape[1]]
    coords = np.column_stack((yy.ravel(), xx.ravel()))

    path = Path(hull_coords)
    inside = path.contains_points(coords)
    hull_mask = inside.reshape(tissue.shape)

    len_surf = len_edge(tissue, res=res)
    len_hull = len_edge(hull_mask, res=res)

    return len_surf/len_hull, len_surf, len_hull

# donut, res = get_picture2D('donut.tif')
# point, _ = get_picture('point.tif')
# height, width = donut.shape
# print(width, height)

# donut_mask = donut != 0

# # we should have only ventricle and outside regions
# labeled, n = label(~ donut_mask)

# if n != 2:
#     raise ValueError('problem with masks, too many regions')

# # outside label is the label of the pixel on the top left
# outside_label = labeled[(0,0)]
# outside_mask = labeled == outside_label
# # ventricle is the other region
# ventr_mask = (labeled != outside_label) & (~ donut_mask)


# # show_pic(donut_mask, 'donut')

# # calculating distance from ventricle
# distance = np.zeros_like(donut, dtype=np.float64)
# # keeping it only in the area of the donut
# distance[donut_mask] = distance_transform_edt(~ventr_mask, sampling= res)[donut_mask]

# # max value in the distance pic
# max_dist = np.max(distance)


# # find center of the ventricle
# ventr_coords = center_of_mass(ventr_mask)


# # find point in the reference pic
# point_mask = point != 0
# point_coords = center_of_mass(point_mask)


# dx = point_coords[0] - ventr_coords[0]
# dy = point_coords[1] - ventr_coords[1]
# angle_0 = math.atan2(dy, dx)

# max_length = np.linalg.norm(donut.shape)
# # how many angles
# n_angles = 40

# results = pd.DataFrame(columns = ['angle_r', 'angle_d', 'thickness', 'ext_x', 'ext_y', 'vent_x', 'vent_y'])

# test_img = distance.copy()

# for i in range(n_angles):

#     # angle of the ray
#     a = 2*math.pi / n_angles * i + angle_0
    
#     line_pixels = get_ray_pixels(donut, point_coords, a, res)

#     cols = line_pixels[:,0]
#     rows = line_pixels[:,1]



#     # add radial picture line on top of picture
#     # test_img[rows, cols] = max_dist

#     # find the max ventricle distance along line
#     current_thickness = np.max(distance[rows, cols])
#     # where is the max distance along line
#     current_index = np.argmax(distance[rows, cols])
#     current_coords = [int(cols[current_index]), int(rows[current_index])]

#     # circe centered in current point
#     circ_pixels = get_circle_pixels(distance, current_coords, current_thickness, res)
#     rows = circ_pixels[:,0]
#     cols = circ_pixels[:,1]

#     # add circle on top of picture
#     # test_img[rows, cols] = max_dist

#     # I need to set the pixels outside the tissue at some max value
#     fake_distance = distance.copy()
#     fake_distance[outside_mask] = max_dist
#     vent_index = np.argmin(fake_distance[rows, cols])
#     vent_coords = [int(cols[vent_index]), int(rows[vent_index])]

#     r_line, c_line = get_line_pixels(current_coords, vent_coords)
#     test_img[r_line, c_line] = max_dist

#     results.loc[len(results)] = [a-angle_0, np.degrees(a-angle_0), current_thickness, *current_coords, *vent_coords]

# # print(results)

# plt.imshow(test_img, cmap = 'magma')
# plt.scatter(results['ext_x'], results['ext_y'], s = 10, color = 'cyan')
# plt.scatter(results['vent_x'], results['vent_y'], s = 10, color = 'cyan')
# plt.show()

# results.to_csv('result.csv')