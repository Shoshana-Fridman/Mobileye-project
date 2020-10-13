from math import sqrt
import numpy as np


def calc_TFL_dist(prev_container, curr_container, focal, pp):
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)
    if abs(tZ) < 10e-6:
        print('tz = ', tZ)
    elif norm_prev_pts.size == 0:
        print('no prev points')
    elif norm_prev_pts.size == 0:
        print('no curr points')
    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, curr_container.valid = calc_3D_data(
            norm_prev_pts, norm_curr_pts, R, foe, tZ)
    return curr_container


def prepare_3D_data(prev_container, curr_container, focal, pp):
    norm_prev_pts = normalize(prev_container.traffic_light, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_light, focal, pp)
    np.array(curr_container)
    R, foe, tZ = decompose(np.array(curr_container.EM))
    return norm_prev_pts, norm_curr_pts, R, foe, tZ


def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ):
    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = []
    corresponding_ind = []
    validVec = []
    for p_curr in norm_curr_pts:
        corresponding_p_ind, corresponding_p_rot = find_corresponding_points(p_curr, norm_rot_pts, foe)
        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0)
        if not valid:
            Z = 0
        validVec.append(valid)
        P = Z * np.array([p_curr[0], p_curr[1], 1])
        pts_3D.append((P[0], P[1], P[2]))
        corresponding_ind.append(corresponding_p_ind)
    return corresponding_ind, np.array(pts_3D), validVec


def normalize(pts, focal, pp):
    # transform pixels into normalized pixels using the focal length and principle point
    normal_pts = np.zeros(shape=pts.shape)
    for i in range(pts.shape[0]):
        normal_pts[i] = [(pts[i][0] - pp[0]) / focal, (pts[i][1] - pp[1]) / focal]
    return normal_pts


def unnormalize(pts, focal, pp):
    # transform normalized pixels into pixels using the focal length and principle point
    unnormal_pts = []
    for i, pt in enumerate(pts):
        unnormal_pts.append([pt[0] * focal + pp[0], pt[1] * focal + pp[1]])
        print(i)
    return np.array(unnormal_pts)


def decompose(EM):
    # extract R, foe and tZ from the Ego Motion
    R = EM[:3, :3]
    T = EM[:3, 3]
    tZ = T[2]
    foe = [T[0] / tZ, T[1] / tZ]
    return R, foe, tZ


def rotate(pts, R):
    # rotate the points - pts using R
    rotate_pts = np.zeros(shape=(pts.shape[0], 3))
    for i in range(pts.shape[0]):
        pt = np.append(pts[i], 1)
        p_r = R.dot(pt)
        rotate_pts[i] = (1 / p_r[2]) * p_r
    return rotate_pts


def get_distance(p, foe, point):
    m = (foe[1] - p[1]) / (foe[0] - p[0])
    n = (p[1] * foe[0] - foe[1] * p[0]) / (foe[0] - p[0])
    return abs((m * point[0] + n - point[1]) / sqrt(m ** 2 + 1))


def find_corresponding_points(p, norm_pts_rot, foe):
    # compute the epipolar line between p and foe
    # run over all norm_pts_rot and find the one closest to the epipolar line
    # return the closest point and its index
    min_distance = 10000
    min_index = -1
    closest_point = []
    for i, point in enumerate(norm_pts_rot):
        curr_distance = get_distance(p, foe, point)
        if min_distance > curr_distance:
            min_distance = curr_distance
            closest_point = point
            min_index = i
    return min_index, closest_point


def get_min_divided(p_curr, p_rot):
    return 0 if p_curr[0] - p_rot[0] < p_curr[1] - p_rot[1] else 1


def calculate_distance(tZ, foe, rot, curr):
    return (tZ * (foe - rot)) / (curr - rot)


def calculate_diffrence(rot, curr):
    return abs(rot - curr)


def calc_dist(p_curr, p_rot, foe, tZ):
    # calculate the distance of p_curr using x_curr, x_rot, foe_x and tZ
    # calculate the distance of p_curr using y_curr, y_rot, foe_y and tZ
    # combine the two estimations and return estimated Z
    Zx = calculate_distance(tZ, foe[0], p_rot[0], p_curr[0])
    Zy = calculate_distance(tZ, foe[1], p_rot[1], p_curr[1])
    x_diff = calculate_diffrence(p_rot[0], p_curr[0])
    y_diff = calculate_diffrence(p_rot[1], p_curr[1])

    return (x_diff * Zx + y_diff * Zy) / (x_diff + y_diff)
