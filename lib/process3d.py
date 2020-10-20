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
from . image_depth import ImageDepth
from cpp.pose_graph import optimize_pose_graph_with_matches, optimize_pose_graph_with_odometry

def process3d(args):
    if args.method >= 4:
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

        if args.view_only:
            img = cv.cvtColor(obj.img, cv.COLOR_RGB2BGR)
            depth = np.maximum(0.0, obj.depth_map)
            depth = (depth / np.max(depth)*255).astype(np.uint8)
            depth = cv.cvtColor(depth, cv.COLOR_GRAY2BGR)

            canvas = cv.hconcat([img, depth])
            cv.imshow(obj.image_file, canvas)

            key = cv.waitKey(0)
            if key == 27: # ESCAPE
                return

            cv.destroyWindow(obj.image_file)

    if args.view_only:
        return

    if args.method == 0:
        camera_edges = sequential_ICP(args, point_clouds, False)
    elif args.method == 1:
        camera_edges = sequential_ICP(args, point_clouds, True)
    if args.method == 2:
        camera_edges = vision_based_registration(args, point_clouds, False)
    elif args.method == 3:
        camera_edges = vision_based_registration(args, point_clouds, True)

    global_pcd = None

    for i, pc in enumerate(point_clouds):
        pc.pcd.transform(pc.pose)
        if global_pcd is None:
            global_pcd = pc.pcd
        else:
            global_pcd += pc.pcd

    cameras = create_cameras(point_clouds, camera_edges)

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
            custom_draw_geometry([mesh] + cameras)
    else:
        # save the points
        # remove normals to save space
        #empty_array = np.zeros((1,3), dtype=np.float64)
        #global_pcd.normals = o3d.utility.Vector3dVector(empty_array)

        o3d.io.write_point_cloud(args.output, global_pcd)

        if args.viz:
            custom_draw_geometry([global_pcd] + cameras)

def match_features(args, pc_i, pc_j):
    i_pts, j_pts = find_sift_matches(pc_i.kp, pc_i.desc, pc_j.kp, pc_j.desc)

    # geometric filter
    _, mask = cv.findFundamentalMat(np.array(i_pts), np.array(j_pts), cv.FM_RANSAC, 3.0)

    if mask is None:
        return None, None, None, None, None, None

    mask = mask.squeeze()
    i_pts = np.array(i_pts)[np.where(mask)]
    j_pts = np.array(j_pts)[np.where(mask)]

    # find common good points
    i_3d, i_2d, i_good_idx = pc_i.project3d(i_pts)
    j_3d, j_2d, j_good_idx = pc_j.project3d(j_pts)
    good_idx = np.intersect1d(i_good_idx, j_good_idx)

    if len(good_idx) == 0:
        return None, None, None, None, None, None

    i_3d = i_3d[good_idx]
    i_2d = i_2d[good_idx]
    j_3d = j_3d[good_idx]
    j_2d = j_2d[good_idx]

    # filter out points with bad alignment
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(i_3d))
    dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(j_3d))

    corr = np.zeros((len(i_3d), 2), dtype=np.int32)
    corr[:,0] = np.arange(0, len(i_3d))
    corr[:,1] = corr[:,0]

    ret = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        src,
        dst,
        o3d.utility.Vector2iVector(corr),
        args.max_point_dist);

    corr = np.array(ret.correspondence_set)

    i_3d = i_3d[corr[:,0]]
    i_2d = i_2d[corr[:,0]]
    j_3d = j_3d[corr[:,0]]
    j_2d = j_2d[corr[:,0]]

    return i_2d, i_3d, j_2d, j_3d, ret.transformation, ret.inlier_rmse

def vision_based_registration(args, point_clouds, pose_graph_optimzation):
    sift = cv.SIFT_create()
    camera_edges = []

    # run SIFT on all the images
    for cur in point_clouds:
        cur.kp, cur.desc = sift.detectAndCompute(cur.gray_undistort, cur.mask)
        print(f"Running SIFT on {cur.image_file}, features={len(cur.kp)}")

    all_matches = dict()

    # match neighbouring pairs
    for i in range(0, len(point_clouds)-1):
        j = i+1

        camera_edges.append((i, j))

        all_matches[i] = dict()
        pc_i = point_clouds[i]
        pc_j = point_clouds[j]

        i_2d, i_3d, j_2d, j_3d, transform, rmse = match_features(args, pc_i, pc_j)

        all_matches[i][j] = (i_2d, i_3d, j_2d, j_3d, transform)

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
            cv.waitKey(20)

        if len(i_2d) < args.min_matches or rmse > args.max_vision_rmse:
            sys.exit("Not enough matches! Try an ICP based method or loosen --max_vision_rmse.")
            all_matches[i][j] = (None, None, None, None, None)

    # initialize the camera poses
    print("\nInitializing camera poses")
    for i in range(1, len(point_clouds)):
        prev = point_clouds[i-1]
        cur = point_clouds[i]

        delta_pose = np.eye(4, 4)

        if (i-1) in all_matches and i in all_matches[i-1]:
            prev_2d, prev_3d, cur_2d, cur_3d, transform = all_matches[i-1][i]

            delta_pose = np.linalg.inv(transform);
            print(f"{cur.image_file}, matches: {len(prev_3d)}")
        else:
            print(f"No matches for {prev.image_file} and {cur.image_file}, using identity pose")

        cur.pose = prev.pose @ delta_pose

    if pose_graph_optimzation:
        # find loop closure (if any) with the last image
        print("\nFinding loop closure")
        all_matches[len(point_clouds)-1] = dict()

        best_num_match = 0
        best_match = None
        best_match_idx = 0

        for j in range(0, min(len(point_clouds)-1, args.loop_closure_range)):
            pc_i = point_clouds[-1]
            pc_j = point_clouds[j]
            i_2d, i_3d, j_2d, j_3d, transform, rmse = match_features(args, pc_i, pc_j)

            if i_2d is None:
                print(f"Matching {pc_i.image_file} {pc_j.image_file}, matches: 0")
                continue

            print(f"Matching {pc_i.image_file} {pc_j.image_file}, matches: {len(i_2d)}, rmse: {rmse:.4f}")

            if len(i_2d) < args.min_matches or rmse > args.max_vision_rmse:
                continue

            if len(i_2d) > best_num_match:
                best_num_match = len(i_2d)
                best_match = (i_2d, i_3d, j_2d, j_3d, transform)
                best_match_idx = j

        print(f"Best loop closure match {point_clouds[best_match_idx].image_file} matches: {best_num_match}")
        all_matches[len(point_clouds)-1][best_match_idx] = best_match
        camera_edges.append((len(point_clouds)-1, best_match_idx))

        if best_num_match == 0:
            sys.exit("Not enough matches for loop closure!")

        print("\nRunning pose graph optimization")
        # poses - these variables get optimized
        poses = np.zeros((len(point_clouds), 7))
        for i, p in enumerate(point_clouds):
            poses[i,:] = get_pose_vector(p.pose)

        # matches - these are the constraints
        matches = np.zeros((0, 8))
        for idx1 in all_matches:
            for idx2 in all_matches[idx1]:
                i_2d, i_3d, j_2d, j_3d, _ = all_matches[idx1][idx2]

                for p, c in zip(i_3d, j_3d):
                    r = np.array([idx1, idx2, p[0], p[1], p[2], c[0], c[1], c[2]])
                    matches = np.vstack((matches, r))

        optim_poses = optimize_pose_graph_with_matches(poses, matches)

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

    return camera_edges

def sequential_ICP(args, point_clouds, pose_graph_optimzation):
    camera_edges = []
    ys = []
    zs = []

    # http://www.open3d.org/docs/release/tutorial/pipelines/multiway_registration.html
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

    for i in range(1, len(point_clouds)):
        prev = point_clouds[i-1]
        cur = point_clouds[i]

        camera_edges.append((i-1, i))

        transform, information = pairwise_ICP_registration(prev.pcd, cur.pcd, args.max_point_dist)
        odometry = transform.transformation @ odometry

        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(
                np.linalg.inv(odometry)))

        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                i-1, i,
                transform.transformation,
                information,
                uncertain=False))

        print(f"sequential ICP: {prev.image_file} -> {cur.image_file}, rmse: {transform.inlier_rmse:.4f}")

        cur.pose = pose_graph.nodes[-1].pose.copy()

        ys.append(cur.pose[1,3])
        zs.append(cur.pose[2,3])

    #plt.plot(ys,zs,marker="o")
    #ax = plt.gca()
    #ax.set_aspect("equal")
    #plt.show()

    if pose_graph_optimzation:
        print("\nFinding loop closure")

        best_rmse = 100000.0
        best_match_idx = 0
        best_transform = None
        best_information = None

        for j in range(0, min(len(point_clouds)-1, args.loop_closure_range)):
            last = point_clouds[-1]
            cur = point_clouds[j]

            transform, information = pairwise_ICP_registration(last.pcd, cur.pcd, args.max_point_dist)

            print(f"Loop closure test: {last.image_file} -> {cur.image_file}, rmse: {transform.inlier_rmse:.4f}, fitness: {transform.fitness:.4f}")

            if transform.inlier_rmse < best_rmse and transform.fitness > 0:
                best_rmse = transform.inlier_rmse
                best_transform = transform.transformation
                best_information = information
                best_match_idx = j

        print(f"Best loop closure match {point_clouds[best_match_idx].image_file} rmse: {best_rmse}")
        camera_edges.append((len(point_clouds)-1, best_match_idx))

        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                len(point_clouds)-1,
                best_match_idx,
                best_transform,
                best_information,
                uncertain=True))

        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=args.max_point_dist,
            edge_prune_threshold=0.25,
            reference_node=0)

        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            o3d.pipelines.registration.global_optimization(
                pose_graph,
                o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                option)

        ys2 = []
        zs2 = []
        for i in range(len(point_clouds)):
            point_clouds[i].pose = pose_graph.nodes[i].pose

            ys2.append(point_clouds[i].pose[1,3])
            zs2.append(point_clouds[i].pose[2,3])

        #plt.plot(ys,zs,marker="o")
        #plt.plot(ys2,zs2,marker="o")
        #ax = plt.gca()
        #ax.set_aspect("equal")
        #plt.show()

    return camera_edges

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

def get_pose_vector(pose):
    r = Rotation.from_matrix(pose[0:3,0:3])
    r = r.as_quat()

    qx = r[0]
    qy = r[1]
    qz = r[2]
    qw = r[3]

    tx = pose[0,3]
    ty = pose[1,3]
    tz = pose[2,3]

    return np.array([qw, qx, qy, qz, tx, ty, tz])

def pairwise_ICP_registration(source, target, icp_dist):
    t = o3d.pipelines.registration.registration_icp(
        source,
        target,
        icp_dist,
        np.eye(4, 4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source,
        target,
        icp_dist,
        t.transformation)

    return t, info

def create_cameras(point_clouds, camera_edges):
    camera_width = 0.05 # world unit
    camera_color = np.array([0.8, 0.8, 0.8])
    camera_edge_color = np.array([0.5, 0.5, 0.8])

    cameras = []

    for pc in point_clouds:
        f = pc.intrinsic[0,0]
        h = pc.img.shape[0]
        w = pc.img.shape[1]

        camera_focal = camera_width*f/w
        camera_height = camera_width*h/w

        vert = np.array([
            [0, 0, 0],
            [ camera_width/2,  camera_height/2, camera_focal],
            [ camera_width/2, -camera_height/2, camera_focal],
            [ -camera_width/2,-camera_height/2,  camera_focal],
            [ -camera_width/2, camera_height/2,  camera_focal],
            ])

        tri_idx = np.array([
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1],
            ])

        camera = o3d.geometry.LineSet(o3d.utility.Vector3dVector(vert), o3d.utility.Vector2iVector(tri_idx))
        camera.transform(pc.pose)
        camera.paint_uniform_color(camera_color)

        cameras.append(camera)

    for camera_edge in camera_edges:
        i, j = camera_edge

        pose_i = point_clouds[i].pose
        pose_j = point_clouds[j].pose

        vert = np.array([
            [pose_i[0,3], pose_i[1,3], pose_i[2,3]],
            [pose_j[0,3], pose_j[1,3], pose_j[2,3]],
            ])

        idx = np.array([
            [0, 1]
            ])

        edge = o3d.geometry.LineSet(o3d.utility.Vector3dVector(vert), o3d.utility.Vector2iVector(idx))
        edge.paint_uniform_color(camera_edge_color)

        cameras.append(edge)

    return cameras