import open3d as o3d
import numpy as np
import cv2 as cv
import json
import struct
import base64

class ImageDepth:
    def __init__(self,
        calibration_file,
        image_file,
        depth_file,
        width=640,
        height=480,
        max_distance=0.5,
        distance_threshold=0.1,
        normal_radius=0.1):

        self.image_file = image_file
        self.calibration_file = calibration_file
        self.depth_file = depth_file
        self.width = width
        self.height = height
        self.max_distance = max_distance
        self.distance_threshold = distance_threshold
        self.normal_radius = normal_radius

        self.load_calibration(calibration_file)
        #self.create_undistortion_lookup()

        self.load_image(image_file)
        self.load_depth(depth_file, max_distance)

    def load_calibration(self, file):
        with open(file) as f:
            data = json.load(f)

            lensDistortionLookupBase64 = data["lensDistortionLookup"]
            inverseLensDistortionLookupBase64 = data["inverseLensDistortionLookup"]
            lensDistortionLookupBinary = base64.decodebytes(lensDistortionLookupBase64.encode("ascii"))
            inverseLensDistortionLookupBinary = base64.decodebytes(inverseLensDistortionLookupBase64.encode("ascii"))
            lensDistortionLookup = []
            inverseLensDistortionLookup = []

            # create arrays of 32-bit floating point numbers
            for i in range(0, len(lensDistortionLookupBinary), 4):
                lensDistortionLookup.append(struct.unpack('<f', lensDistortionLookupBinary[i:i+4])[0])
            for i in range(0, len(inverseLensDistortionLookupBinary), 4):
                inverseLensDistortionLookup.append(struct.unpack('<f', inverseLensDistortionLookupBinary[i:i+4])[0])

            self.lensDistortion = lensDistortionLookup
            self.inverseLensDistortion = inverseLensDistortionLookup

            self.intrinsic = np.array(data["intrinsic"]).reshape((3,3))
            self.intrinsic = self.intrinsic.transpose()

            self.scale = float(self.width) / data["intrinsicReferenceDimensionWidth"]
            self.intrinsic[0,0] *= self.scale
            self.intrinsic[1,1] *= self.scale
            self.intrinsic[0,2] *= self.scale
            self.intrinsic[1,2] *= self.scale

    def create_undistortion_lookup(self):
        xy = [(x,y) for y in range(0, self.height) for x in range(0, self.width)]
        xy = np.array(xy, dtype=np.float32).reshape(-1,2)

        # subtract center
        xy -= self.intrinsic[0:1, 2]

        # calc radius from center
        r = np.sqrt(xy[:,0]**2 + xy[:,1]**2)

        # normalize radius
        max_r = np.max(r)
        norm_r = r / max_r

        # interpolate the magnification
        # this seems really silly ...
        num = len(self.inverseLensDistortion)
        magnification = np.interp(norm_r*num, np.arange(0, num), self.inverseLensDistortion)

        print(self.inverseLensDistortion)
        print(magnification)
        # unit vector
        magnititude = np.expand_dims(np.sqrt(xy[:,0]**2 + xy[:,1]**2), 1)
        unit_xy = xy / magnititude

        new_xy = unit_xy * magnititude * np.expand_dims(magnification, 1)
        print(new_xy)
        #print(xy.shape)

    def load_depth(self, file, max_distance):
        depth = np.fromfile(file, dtype='float16')

        # vectorize version, faster
        # all possible (x,y) position
        xyz = [(x,y,1.0) for y in range(0, self.height) for x in range(0, self.width)]
        xyz = np.array(xyz)

        # remove bad values
        no_nan = np.invert(np.isnan(depth))
        good_range = depth < max_distance
        idx = np.logical_and(no_nan, good_range)
        xyz = xyz[np.where(idx)]

        # mask out depth buffer
        self.mask = idx
        self.depth_map = depth
        self.depth_map[np.where(idx == False)] = 0
        self.depth_map = self.depth_map.reshape((self.height, self.width, 1))

        per = float(np.sum(idx==True))/len(depth)
        print(f"Processing {file}, keeping={np.sum(idx==True)}/{len(depth)} ({per:.3f}) points")

        depth = np.expand_dims(depth[np.where(idx)],1)

        # project to 3D
        xyz = np.linalg.inv(self.intrinsic) @ xyz.transpose()

        # apply depth
        xyz = xyz.transpose() * (depth*3)

        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(xyz)

        # calc normal, required for ICP point-to-plane
        self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.normal_radius, max_nn=10))

    def load_image(self, file):
        img = np.fromfile(file, dtype='uint8')
        img = img.reshape((self.height, self.width, 4))
        img = img[:,:,0:3]

        # swap RB
        self.img = img[:,:,[2,1,0]]
        self.gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    def preprocess_point_cloud(self, pcd, voxel_size):
        #print("  Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 2
        #print("  Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        #print("  Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

        return pcd_down, pcd_fpfh

    def get_point3d(self, x, y):
        fx = self.intrinsic[0,0]
        fy = self.intrinsic[1,1]
        cx = self.intrinsic[0,2]
        cy = self.intrinsic[1,2]

        xf = (x - cx)/fx
        yf = (y - cy)/fy

        depth = self.depth_map[y, x]

        if depth > 0:
            return np.array([xf*depth, yf*depth, depth]).reshape(1, 3)

        return None