import sys
import glob
import copy
import math
import time
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2 as cv
from mpl_toolkits.axes_grid1 import ImageGrid
from . rigid_transform_3D import rigid_transform_3D
from . image_depth import ImageDepth

def process3d(args):
    image_files = sorted(glob.glob(f"{args.folder}/video*.bin"))
    depth_files = sorted(glob.glob(f"{args.folder}/depth*.bin"))
    calibration_file =f"{args.folder}/calibration.json"

    if len(image_files) == 0:
        print("No image files found")
        sys.exit(0)

    if len(depth_files) == 0:
        print("No depth files found")
        sys.exit(0)

    # generate some colors for the point cloud
    val = np.arange(len(depth_files)) / len(depth_files)
    colors = plt.cm.jet(val)
    colors = colors[:, 0:3]

    # final global point cloud
    global_pcd = None
    point_clouds = []

    # load all the point clouds into memory
    for i, (image_file, depth_file) in enumerate(zip(image_files, depth_files)):
        obj = ImageDepth(
            calibration_file,
            image_file,
            depth_file,
            args.width,
            args.height,
            args.max_distance,
            args.distance_threshold,
            args.normal_radius)

        obj.pcd.paint_uniform_color(colors[i])

        point_clouds.append(obj)

    # run sequential registration
    prev = point_clouds.pop(0)
    global_pcd = prev.pcd
    last_transform = np.eye(4,4)
    last_vel = np.eye(4,4)

    for cur in point_clouds:
        print(cur.image_file, cur.depth_file)

        if args.vis_tracking:
            sift = cv.SIFT_create()
            kp1, desc1 = sift.detectAndCompute(prev.gray, None)
            kp2, desc2 = sift.detectAndCompute(cur.gray, None)

            MIN_MATCH_COUNT = 5
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks = 50)

            flann = cv.FlannBasedMatcher(index_params,search_params)

            knn_matches = flann.knnMatch(desc1, desc2, k=2)

            pt1 = []
            pt2 = []

            for m in knn_matches:
                kp1_idx = m[0].queryIdx

                #if kp1_idx in good_kp1_idx:
                dist1 = m[0].distance
                dist2 = m[1].distance

                if dist1 < dist2*0.7:
                    kp2_idx = m[0].trainIdx

                    pt1.append(kp1[kp1_idx].pt)
                    pt2.append(kp2[kp2_idx].pt)

            _, mask = cv.findFundamentalMat(np.array(pt1), np.array(pt2), cv.FM_RANSAC, 3.0)

            mask = mask.squeeze()
            pt1 = np.array(pt1)[np.where(mask)]
            pt2 = np.array(pt2)[np.where(mask)]

            prev_3d = np.zeros((0,3))
            cur_3d = np.zeros((0,3))
            good_idx = []

            t = time.time()
            for idx, (p1, p2) in enumerate(zip(pt1, pt2)):
                prev_x = int(p1[0] + 0.5)
                prev_y = int(p1[1] + 0.5)

                cur_x = int(p2[0] + 0.5)
                cur_y = int(p2[1] + 0.5)

                p3d = prev.get_point3d(prev_x, prev_y)
                c3d = cur.get_point3d(cur_x, cur_y)

                if p3d is not None and c3d is not None:
                    prev_3d = np.vstack((prev_3d, p3d))
                    cur_3d = np.vstack((cur_3d, c3d))
                    good_idx.append(idx)

            R, t = rigid_transform_3D(cur_3d.transpose(), prev_3d.transpose())
            delta = np.eye(4, 4)
            delta[0:3, 0:3] = R
            delta[0:3, 3:4] = t
            print("delta")
            print(delta)

            img = cv.cvtColor(cur.gray, cv.COLOR_GRAY2BGR)

            pt1 = pt1[good_idx]
            pt2 = pt2[good_idx]

            for a, b in zip(pt1, pt2):
                aa = (int(a[0].item()), int(a[1].item()))
                bb = (int(b[0].item()), int(b[1].item()))
                img = cv.line(img, aa, bb, (0,0,255))

            cv.imshow("cur", img)
            cv.waitKey(100)

        guess = last_transform# @ delta
        reg = o3d.registration.registration_icp(
            cur.pcd,
            global_pcd,
            cur.distance_threshold,
            guess,
            o3d.registration.TransformationEstimationPointToPlane())

        # apply transform and merge point cloud
        cur.pcd.transform(reg.transformation)
        global_pcd += cur.pcd

        # estimate crude velocity
        last_vel = np.linalg.inv(last_transform) @ reg.transformation
        print("last_vel")
        print(last_vel)
        print("")

        last_transform = reg.transformation
        prev = cur

    # save the points
    # remove normals to save space
    #empty_array = np.zeros((1,3), dtype=np.float64)
    #global_pcd.normals = o3d.utility.Vector3dVector(empty_array)

    print(f"Saving to {args.output} ...")
    o3d.io.write_point_cloud(args.output, global_pcd)

    if args.viz:
        def custom_draw_geometry(pcd, name="Open3D"):
            vis = o3d.visualization.Visualizer()
            vis.create_window(name)

            for p in pcd:
                vis.add_geometry(p)

            opt = vis.get_render_option()
            opt.background_color = np.asarray([0, 0, 0])
            opt.point_size = 1.0

            vis.run()
            vis.destroy_window()

        custom_draw_geometry([global_pcd])
