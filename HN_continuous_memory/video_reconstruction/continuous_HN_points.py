
import torch
import numpy as np
import argparse
import torch.nn.functional as F
from PIL import Image
import sys
import os

current_dir = os.getcwd()
sys.path.append(current_dir)
import json
from utils import (load_video, DiscreteHopfieldNet, ContinuousHopfieldNet)

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    
    # Make all arguments required
    parser.add_argument("--video-folder", required=True, help="Path to video folder.")
    # New arguments based on constants
    parser.add_argument("--N", type=int, required=True, help="memory size")
    parser.add_argument("--nb_basis", type=int, required=True, help="number of basis")
    parser.add_argument("--resolution", type=int, required=True, help="resolution")
    parser.add_argument("--num_iters", type=int, required=True, help="number of Hopfield iters")
    parser.add_argument("--beta", type=float, required=True, help="Hopfield beta")
    parser.add_argument("--mask", type=float, required=True, help="mask percentage")
    parser.add_argument("--num_points", nargs="+", type=int, required=True, help="mask percentage")
    parser.add_argument(
        "--options",
        nargs="+",
        help="Override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file (deprecated)."
    )
    
    
    args = parser.parse_args()
    return args

def main(args):
    beta = args.beta 
    num_iters = args.num_iters
    N = args.N
    resolution = args.resolution
    output_file = f"video_reconstruction/results/beta_{args.beta}_nb_basis_{args.nb_basis}_num_iters_{args.num_iters}_N_{args.N}_resolution_{args.resolution}_mask_{args.mask}_num_points_{args.num_points}.json"
    results = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            results = json.load(f)
    for video_file in os.listdir(args.video_folder):
        if video_file.endswith(".mp4") and video_file not in results:
            video_path = os.path.join(args.video_folder, video_file)
            # Select the first N samples
            video = load_video(video_path, num_segments=N, resolution=resolution)
            video = video.reshape(N, 3, resolution, resolution)
            mask = torch.ones_like(video)  # Start with a tensor of ones
            mask[:, :, int(args.resolution*(1-args.mask)):, :] = 0  # Zero out the first half of the height (224 // 2 = 112)

            # Apply the mask to the video tensor
            video_masked = video * mask
            X = video.view(N, -1).to("cuda")
            Q = video_masked.view(N,-1).to("cuda")
            model_discrete = DiscreteHopfieldNet(beta, num_iters, "cuda")
            pred= model_discrete(X, Q).to("cpu")
            pred = F.normalize(pred, p=2, dim=1)
            X_norm = F.normalize(X, p=2, dim=1).to("cpu")
            sim_discrete = (pred * X_norm).sum(dim=-1).mean().item()
            sims_cont = []
            for num_points in args.num_points:
                model = ContinuousHopfieldNet(beta, args.nb_basis, num_iters, num_points, "cuda")
                pred = model(X, Q).to("cpu")
                pred = F.normalize(pred, p=2, dim=1)
                sims_cont.append((pred * X_norm).sum(dim=1).mean().item())
            results[video_file] = {"discrete": sim_discrete, "continuous": sims_cont}
            
            if os.path.exists(output_file):
                os.remove(output_file)  # Optional: write an empty list to reset the file

            # Append new results
            with open(output_file, 'a') as output_json_file:
                json.dump(results, output_json_file)
if __name__ == '__main__':
    args = parse_args()
    main(args)