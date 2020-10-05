import argparse
from lib.process3d import process3d

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TrueDepth camera point cloud registration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('folder', help='folder containing bins and camera calibration')
    parser.add_argument('--viz', type=bool, default=True, help='visualize result')
    parser.add_argument('--output', default="output.ply", help='save PLY file')
    parser.add_argument('--width', type=int, default=640, help='image width')
    parser.add_argument('--height', type=int, default=480, help='image height')
    parser.add_argument('--min_depth', type=float, default=0.1, help='min depth distance (meters)')
    parser.add_argument('--max_depth', type=float, default=0.5, help='max depth distance (meters)')
    parser.add_argument('--distance_threshold', type=float, default=0.02, help='for ICP (meters)')
    parser.add_argument('--normal_radius', type=float, default=0.1, help='max radius for normal calculation (meters)')
    parser.add_argument('--vis_tracking', type=bool, default=True, help='visual tracking, use RGB images to help with registration')

    args = parser.parse_args()

    process3d(args)