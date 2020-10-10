import argparse
from lib.process3d import process3d

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TrueDepth camera point cloud registration',
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('folder', help='folder containing bins and camera calibration')
    parser.add_argument('--viz', type=int, default=1, help='visualize result')
    parser.add_argument('--output', default="output.ply", help='save PLY file')
    parser.add_argument('--width', type=int, default=640, help='image width')
    parser.add_argument('--height', type=int, default=480, help='image height')
    parser.add_argument('--min_depth', type=float, default=0.1, help='min depth distance')
    parser.add_argument('--max_depth', type=float, default=0.5, help='max depth distance')

    parser.add_argument('--method', type=int, default=0, help='Registration method\n0: global vision based (2d/3d with pose graph optimization)\n1: sequential vision based (2d/3d)\n2: sequential ICP (3d only)')
    parser.add_argument('--distance_threshold', type=float, default=0.02, help='max distance between points for ICP methods')
    parser.add_argument('--normal_radius', type=float, default=0.1, help='max radius for normal calculation for ICP methods')
    parser.add_argument('--min_matches', type=int, default=100, help='min matches for vision based method')
    parser.add_argument('--max_look_ahead', type=int, default=10, help='max images to look ahead for vision based method, set -1 to match all (SLOW!)')
    parser.add_argument('--uniform_color', type=int, default=0, help='use uniform color for point instead of RGB image')


    args = parser.parse_args()

    process3d(args)