from pathlib import Path
from scipy.io import loadmat
import cv2
import imghdr
from pprint import pprint
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def parse_pts(ann_file):
    # 300W
    ann = []
    with open(ann_file, 'r') as f:
        version = int(f.readline().split(':')[1].strip())
        n_points = int(f.readline().split(':')[1].strip())
        print('ann {}, version is {}, n_points is {}'.format(ann_file, version, n_points))
        _ = f.readline()
        for i in range(n_points):
            x, y = [int(float(x)) for x in f.readline().split(' ')]
            ann.append([x, y])
    return np.array(ann)


# def to_int(arr):
#     return arr.astype(np.int32) if arr is not None else np.empty(0, dtype=np.int32)

to_int = lambda arr: arr.astype(np.int32) if arr is not None else np.empty(0, dtype=np.int32)
transpose = lambda pt: pt.T if pt is not None and pt.shape[0] < pt.shape[1] else pt


def parse_mat(ann_file):
    mat = loadmat(ann_file)
    roi = mat.get('roi')  # 300W-3D
    pt2d = mat.get('pt2d')  # 300W-3D
    if pt2d is None:
        pt2d = mat.get('pts_2d')  # 300W-LP
    pt3d = mat.get('pts_3d')  # 300W-LP
    if pt3d is None:
        pt3d = mat.get('pt3d_68')  # AFLW2000-3D
    if pt3d is None:
        pt3d = mat.get('Fitted_Face')  # 300W-3D-Face

    # if pt2d is not None and pt2d.shape[0] < pt2d.shape[1]:
    #     pt2d = pt2d.T
    #
    # if pt2d is not None and pt2d.shape[0] < pt2d.shape[1]:
    #     pt2d = pt2d.T
    pt2d = transpose(pt2d)
    pt3d = transpose(pt3d)

    roi = to_int(roi)
    pt2d = to_int(pt2d)
    pt3d = to_int(pt3d)

    print('file {}, \nmat info header {}, \n'
          'version {}, globals {}, \n'
          'pt2d shape {}, roi {}, pt3d {}'
          .format(ann_file,
                  mat.get('__header__'),
                  mat.get('__version__'),
                  mat.get('__globals__'),
                  pt2d.shape,
                  roi,
                  pt3d.shape))
    return roi, pt2d, pt3d


GREEN = (0, 255, 0)
RED = (0, 0, 255)
CYAN = (255, 255, 0)


def vis_align(img_dir, ann_dir, vis_type, size=None):
    assert vis_type in ['2d', '3d']
    print('{} {} {} {}'.format(img_dir, ann_dir, vis_type, size))
    if ann_dir is None:
        ann_dir = img_dir
    if size is not None and len(size) == 1:
        size = size * 2
    img_root = Path(img_dir)
    ann_dir = Path(ann_dir)
    img_files = sorted([x for x in img_root.glob('*')
                        if os.path.isfile(x) and imghdr.what(x) is not None])
    ann_files = sorted([x for x in ann_dir.glob('*')
                        if os.path.isfile(x) and imghdr.what(x) is None],
                       key=lambda x: str(x).replace('_pts', ''))

    files = list(zip(img_files, ann_files))
    pprint(files[:2])
    for img_file, ann_file in files:
        print('img_file {}\nann_file {}\n'.format(img_file, ann_file))
        img = cv2.imread(str(img_file))

        if ann_file.suffix == '.pts':
            landmark = parse_pts(ann_file)

            for i, (x, y) in enumerate(landmark):
                cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, CYAN)
                cv2.circle(img, (x, y), 1, GREEN)
        elif ann_file.suffix == '.mat':
            roi, pt2d, pt3d = parse_mat(ann_file)
            for i, pt in enumerate(roi):
                cv2.rectangle(img, (pt[0], pt[1]), (pt[2], pt[3]), RED)
            if len(pt3d) == 0:
                for i, (x, y) in enumerate(pt2d):
                    cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, CYAN)
                    cv2.circle(img, (x, y), 1, GREEN)
            if len(pt3d.shape) > 1 and pt3d.shape[1] == 2:
                for i, (x, y) in enumerate(pt3d):
                    cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, CYAN)
                    cv2.circle(img, (x, y), 1, GREEN)
            if len(pt3d.shape) > 1 and pt3d.shape[1] == 3:
                if pt3d.shape[0] == 53215:
                    img = cv2.flip(img, 0)  # 垂直翻转

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(pt3d[:, 0], pt3d[:, 1], pt3d[:, 2])
                plt.show(block=False)
                for i, (x, y, z) in enumerate(pt3d):
                    cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, CYAN)
                    cv2.circle(img, (x, y), 1, GREEN)
        if size:
            img = cv2.resize(img, size)
        cv2.imshow('Align', img)
        if cv2.waitKey() == ord('q'):
            exit()
        plt.close()
        # ax.cla()
