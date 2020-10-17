import sys
import glob
import copy
import math
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import open3d as o3d
import matplotlib.pyplot as plt
import cv2 as cv
from . rigid_transform_3D import rigid_transform_3D
from . image_depth import ImageDepth
from cpp.pose_graph import pose_graph

def process3d(args):
    if args.method >= 3:
        sys.exit("Unsupported registration method")

    image_files = sorted(glob.glob(f"{args.folder}/video*.bin"))#[0:3]
    depth_files = sorted(glob.glob(f"{args.folder}/depth*.bin"))#[0:3]
    calibration_file =f"{args.folder}/calibration.json"

    if len(image_files) == 0:
        print("No image files found")
        sys.exit(0)

    if len(depth_files) == 0:
        print("No depth files found")
        sys.exit(0)

    # generate some colors for the point cloudi
    if args.uniform_color:
        val = np.arange(len(depth_files)) / len(depth_files)
        colors = plt.cm.jet(val)
        colors = colors[:, 0:3]

    point_clouds = []

    # load data
    for i, (image_file, depth_file) in enumerate(zip(image_files, depth_files)):
        obj = ImageDepth(
            calibration_file,
            image_file,
            depth_file,
            args.width,
            args.height,
            args.min_depth,
            args.max_depth,
            args.normal_radius)

        if args.uniform_color:
            obj.pcd.paint_uniform_color(colors[i])

        point_clouds.append(obj)

        #cv.imshow("img", obj.depth_map)
        #cv.waitKey(0)

    if args.method == 0:
        sequential_ICP(args, point_clouds)
    if args.method == 1:
        vision_based_registration(args, point_clouds, False)
    elif args.method == 2:
        vision_based_registration(args, point_clouds, True)

    global_pcd = None

    for i, pc in enumerate(point_clouds):
        pc.pcd.transform(pc.pose)

        if global_pcd is None:
            global_pcd = pc.pcd
        else:
            global_pcd += pc.pcd

    print(f"Saving to {args.output} ...")

    if args.mesh:
        print("Meshing ...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(global_pcd, depth=args.mesh_depth)

        if args.keep_largest_mesh:
            print("Finding the largest mesh ...")
            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
                triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()

                largest_cluster_idx = np.array(cluster_n_triangles).argmax()
                triangles_to_remove = triangle_clusters != largest_cluster_idx
                mesh.remove_triangles_by_mask(triangles_to_remove)

        o3d.io.write_triangle_mesh(args.output, mesh)

        if args.viz:
            custom_draw_geometry([mesh])
    else:
        # save the points
        # remove normals to save space
        #empty_array = np.zeros((1,3), dtype=np.float64)
        #global_pcd.normals = o3d.utility.Vector3dVector(empty_array)

        o3d.io.write_point_cloud(args.output, global_pcd)

        if args.viz:
            custom_draw_geometry([global_pcd])

def match_features(args, pc_i, pc_j):
    i_pts, j_pts = find_sift_matches(pc_i.kp, pc_i.desc, pc_j.kp, pc_j.desc)

    # geometric constraint
    _, mask = cv.findFundamentalMat(np.array(i_pts), np.array(j_pts), cv.FM_RANSAC, 3.0)

    if mask is None:
        return None, None, None, None, None

    mask = mask.squeeze()
    i_pts = np.array(i_pts)[np.where(mask)]
    j_pts = np.array(j_pts)[np.where(mask)]

    # find common good points
    i_3d, i_2d, i_good_idx = pc_i.project3d(i_pts)
    j_3d, j_2d, j_good_idx = pc_j.project3d(j_pts)
    good_idx = np.intersect1d(i_good_idx, j_good_idx)

    if len(good_idx) == 0:
        return None, None, None, None, None

    i_3d = i_3d[good_idx]
    i_2d = i_2d[good_idx]
    j_3d = j_3d[good_idx]
    j_2d = j_2d[good_idx]

    R, t, rmse = rigid_transform_3D(i_3d.transpose(), j_3d.transpose())

    return i_2d, i_3d, j_2d, j_3d, rmse

def vision_based_registration(args, point_clouds, pose_graph_optimzation):
    sift = cv.SIFT_create()

    # run SIFT on all the images
    for cur in point_clouds:
        cur.kp, cur.desc = sift.detectAndCompute(cur.gray_undistort, cur.mask)
        print(f"Running SIFT on {cur.image_file}, features={len(cur.kp)}")

    all_matches = dict()

    # match neighbouring pairs
    for i in range(0, len(point_clouds)-1):
        j = i+1

        all_matches[i] = dict()
        pc_i = point_clouds[i]
        pc_j = point_clouds[j]

        i_2d, i_3d, j_2d, j_3d, rmse = match_features(args, pc_i, pc_j)

        all_matches[i][j] = (i_2d, i_3d, j_2d, j_3d)

        if i_2d is None:
            print(f"Matching {pc_i.image_file} {pc_j.image_file}, matches: 0")
            continue

        print(f"Matching {pc_i.image_file} {pc_j.image_file}, matches: {len(i_2d)}, rmse: {rmse:.4f}")

        if args.viz:
            img = cv.cvtColor(pc_i.gray_undistort, cv.COLOR_GRAY2BGR)

            # draw matches
            if i_2d is not None:
                for a, b in zip(i_2d, j_2d):
                    aa = (a[0].item(), a[1].item())
                    bb = (b[0].item(), b[1].item())
                    img = cv.line(img, aa, bb, (0,0,255))

            cv.imshow("cur", img)
            cv.waitKey(10)

        if len(i_2d) < args.min_matches or rmse > args.max_vision_rmse:
            sys.exit("Not enough matches! Try increasing the max depth or try an ICP based method.")
            all_matches[i][j] = (None, None, None, None)

    if pose_graph_optimzation:
        # find loop closure (if any) with the last image
        print("\nFinding loop closure")
        all_matches[len(point_clouds)-1] = dict()

        best_num_match = 0
        best_match = None
        best_match_idx = 0

        for j in range(0, min(len(point_clouds), args.loop_closure_range)):
            pc_i = point_clouds[-1]
            pc_j = point_clouds[j]
            i_2d, i_3d, j_2d, j_3d, rmse = match_features(args, pc_i, pc_j)

            if i_2d is None:
                print(f"Matching {pc_i.image_file} {pc_j.image_file}, matches: 0")
                continue

            print(f"Matching {pc_i.image_file} {pc_j.image_file}, matches: {len(i_2d)}, rmse: {rmse:.4f}")

            if len(i_2d) < args.min_matches or rmse > args.max_vision_rmse:
                continue

            if len(i_2d) > best_num_match:
                best_num_match = len(i_2d)
                best_match = (i_2d, i_3d, j_2d, j_3d)
                best_match_idx = j

        print(f"Best loop closure match {point_clouds[best_match_idx].image_file} matches: {best_num_match}")
        all_matches[len(point_clouds)-1][best_match_idx] = best_match

        if best_num_match == 0:
            sys.exit("Not enough matches for loop closure!")

    # run sequential matching to initialize the camera poses
    print("\nInitializing camera poses")
    for i in range(1, len(point_clouds)):
        prev = point_clouds[i-1]
        cur = point_clouds[i]

        delta_pose = np.eye(4, 4)

        prev_2d = None
        cur_2d = None

        if (i-1) in all_matches and i in all_matches[i-1]:
            prev_2d, prev_3d, cur_2d, cur_3d = all_matches[i-1][i]
            R, t, rmse = rigid_transform_3D(cur_3d.transpose(), prev_3d.transpose())

            delta_pose[0:3, 0:3] = R
            delta_pose[0:3, 3:4] = t
            print(f"{cur.image_file}, matches: {len(prev_3d)}")
        else:
            print(f"No matches for {prev.image_file} and {cur.image_file}, using identity pose")

        #print(delta_pose)
        #print("vision rmse:", rmse)

        cur.pose = prev.pose @ delta_pose


    if pose_graph_optimzation:
        print("\nRunning pose graph optimization")

        # poses - these variables get optimized
        poses = np.zeros((len(point_clouds), 7))
        for i, p in enumerate(point_clouds):
            r = Rotation.from_matrix(p.pose[0:3,0:3])
            r = r.as_quat()

            qx = r[0]
            qy = r[1]
            qz = r[2]
            qw = r[3]

            poses[i, 0:4] = np.array([qw, qx, qy, qz])
            poses[i, 4:7] = p.pose[0:3,3]

        # matches - these are the constraints
        matches = np.zeros((0, 8))
        for idx1 in all_matches:
            for idx2 in all_matches[idx1]:
                i_2d, i_3d, j_2d, j_3d = all_matches[idx1][idx2]

                for p, c in zip(i_3d, j_3d):
                    r = np.array([idx1, idx2, p[0], p[1], p[2], c[0], c[1], c[2]])
                    matches = np.vstack((matches, r))

        optim_poses = pose_graph(poses, matches)

        # read back optimize values
        for pc, optim_pose in zip(point_clouds, optim_poses):
            qw, qx, qy, qz = optim_pose[0:4]
            tx, ty, tz = optim_pose[4:7]

            r = Rotation.from_quat([qx, qy, qz, qw])

            pose = np.eye(4,4)
            pose[0:3,0:3] = r.as_matrix()
            pose[0,3] = tx
            pose[1,3] = ty
            pose[2,3] = tz

            pc.pose = pose

def sequential_ICP(args, point_clouds):
    for i in range(1, len(point_clouds)):
        prev = point_clouds[i-1]
        cur = point_clouds[i]

        print(cur.image_file, cur.depth_file)

        delta_pose = np.eye(4, 4)

        guess = np.eye(4,4)
        reg = o3d.registration.registration_icp(
            cur.pcd,
            prev.pcd,
            args.icp_max_dist,
            guess,
            o3d.registration.TransformationEstimationPointToPlane())

        delta_pose = reg.transformation
        cur.pose = prev.pose @ delta_pose

def custom_draw_geometry(pcd, name="Open3D"):
    vis = o3d.visualization.Visualizer()
    vis.create_window(name)

    for p in pcd:
        vis.add_geometry(p)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    vis.add_geometry(axis)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 1.0

    vis.run()
    vis.destroy_window()

def find_sift_matches(prev_kp, prev_desc, cur_kp, cur_desc):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

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

    return prev_pts, cur_pts
