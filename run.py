import argparse
from lib.process3d import process3d

if __name__ == "__main__":
    class ArgumentParserWithDefaults(argparse.ArgumentParser):
        def add_argument(self, *args, help=None, default=None, **kwargs):
            if help is not None:
                kwargs["help"] = help
            if default is not None and args[0] != "-h":
                kwargs["default"] = default
                if help is not None:
                    kwargs["help"] += " (default: {})".format(default)
            super().add_argument(*args, **kwargs)

    parser = ArgumentParserWithDefaults(description="TrueDepth camera point cloud registration",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("folder", help="folder containing bins and camera calibration")
    parser.add_argument("--viz", type=int, default=1, help="visualize result")

    method = "Registration method\n"
    method += "0: sequential ICP\n"
    method += "1: sequential ICP with loop closure\n"
    method += "2: sequential vision based\n"
    method += "3: sequential vision based with loop closure\n"

    parser.add_argument("--method", type=int, default=3, help=method)
    parser.add_argument("--output", default="output.ply", help="save PLY file")
    parser.add_argument("--width", type=int, default=640, help="image width")
    parser.add_argument("--height", type=int, default=480, help="image height")
    parser.add_argument("--min_depth", type=float, default=0.1, help="min depth distance")
    parser.add_argument("--max_depth", type=float, default=0.5, help="max depth distance")
    parser.add_argument("--max_point_dist", type=float, default=0.02, help="max distance between points for ICP/vision methods")
    parser.add_argument("--icp_loop_closure_dist", type=float, default=0.1, help="max distance between points for ICP loop closure")
    parser.add_argument("--normal_radius", type=float, default=0.01, help="max radius for normal calculation for ICP methods")
    parser.add_argument("--min_matches", type=int, default=30, help="min matches for vision based method")
    parser.add_argument("--loop_closure_range", type=int, default=10, help="search N images from the start to find a loop closure with the last image")
    parser.add_argument("--uniform_color", type=int, default=0, help="use uniform color for point instead of RGB image")
    parser.add_argument("--max_vision_rmse", type=float, default=0.04, help="max rmse when estimating pose using vision")
    parser.add_argument("--mesh", type=int, default=0, help="make a mesh instead of point cloud")
    parser.add_argument("--mesh_depth", type=int, default=10, help="Poisson reconstruction depth, higher results in more detail")
    parser.add_argument("--keep_largest_mesh", type=int, default=0, help="keep only the largest mesh, useful for filtering noise")

    args = parser.parse_args()

    process3d(args)