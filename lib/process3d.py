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
    image_files = sorted(glob.glob(f"{args.folder}/video*.bin")) #[18:22]
    depth_files = sorted(glob.glob(f"{args.folder}/depth*.bin")) #[18:22]
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
            args.min_depth,
            args.max_depth,
            args.distance_threshold,
            args.normal_radius)

        obj.pcd.paint_uniform_color(colors[i])

        point_clouds.append(obj)

    # run sequential registration
    prev = point_clouds.pop(0)
    global_pcd = prev.pcd
    last_transform = np.eye(4,4)
    last_pose_delta = np.eye(4,4)

    sift = cv.SIFT_create()
    prev_kp, prev_desc = sift.detectAndCompute(prev.gray_undistort, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    for cur in point_clouds:
        print(cur.image_file, cur.depth_file)

        if args.vis_tracking:
            cur_kp, cur_desc = sift.detectAndCompute(cur.gray_undistort, None)

            knn_matches = flann.knnMatch(prev_desc, cur_desc, k=2)

            prev_pts = []
            cur_pts = []

            for m in knn_matches:
                prev_idx = m[0].queryIdx

                dist1 = m[0].distance
                dist2 = m[1].distance

                # sift ratio test
                if dist1 < dist2*0.7:
                    cur_idx = m[0].trainIdx

                    prev_pts.append(prev_kp[prev_idx].pt)
                    cur_pts.append(cur_kp[cur_idx].pt)

            # geometric constraint
            _, mask = cv.findFundamentalMat(np.array(prev_pts), np.array(cur_pts), cv.FM_RANSAC, 3.0)

            mask = mask.squeeze()
            prev_pts = np.array(prev_pts)[np.where(mask)]
            cur_pts = np.array(cur_pts)[np.where(mask)]

            prev_3d, prev_2d, prev_good_idx = prev.project3d(prev_pts)
            cur_3d, cur_2d, cur_good_idx = cur.project3d(cur_pts)

            # find common good points
            good_idx = np.intersect1d(prev_good_idx, cur_good_idx)

            prev_3d = prev_3d[good_idx]
            prev_2d = prev_2d[good_idx]
            cur_3d = cur_3d[good_idx]
            cur_2d = cur_2d[good_idx]

            R, t, rmse = rigid_transform_3D(cur_3d.transpose(), prev_3d.transpose())
            delta = np.eye(4, 4)
            delta[0:3, 0:3] = R
            delta[0:3, 3:4] = t
            print("vision delta")
            print(delta)
            print("vision rmse:", rmse)
            img = cv.cvtColor(cur.gray_undistort, cv.COLOR_GRAY2BGR)

            for a, b in zip(prev_2d, cur_2d):
                aa = (a[0].item(), a[1].item())
                bb = (b[0].item(), b[1].item())
                img = cv.line(img, aa, bb, (0,0,255))

            if args.viz:
                cv.imshow("cur", img)
                cv.waitKey(1)

        guess = last_transform @ delta
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
        transform_delta = np.linalg.inv(last_transform) @ reg.transformation
        print("depth map delta")
        print(transform_delta)
        print("")

        last_transform = reg.transformation
        prev = cur
        prev_kp = cur_kp
        prev_desc = cur_desc

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

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

        custom_draw_geometry([global_pcd, axis])
