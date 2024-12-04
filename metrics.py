from shapely.geometry import LineString, box 
from shapely import affinity
import numpy as np

def IoU3D(box1, box2):
    # Inputs:
    #     box1 - The first box including (cx, cy, cz, l, w, h, angle)
    #     box2 - The second box including (cx, cy, cz, l, w, h, angle)
    # Outputs:
    #     iou - The result 3D IoU of two boxes
    result_xy, result_z, result_v = [], [], []
    for b in [box1, box2]:
        x, y, z, l, w, h, yaw = b
        result_v.append(l * w * h)
        ls = LineString([[0, z - h/2], [0, z + h/2]])
        result_z.append(ls)
        poly = box(x - l/2, y - w/2, x + l/2, y + w/2)
        poly_rot = affinity.rotate(poly, yaw, use_radians=True)
        result_xy.append(poly_rot)

    overlap_xy = result_xy[0].intersection(result_xy[1]).area
    overlap_z = result_z[0].intersection(result_z[1]).length
    overlap_xyz = overlap_z * overlap_xy
    return overlap_xyz / (np.sum(result_v) - overlap_xyz)