import argparse
import csv
import json
import os
import time
from tqdm import tqdm

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import normalize, resize, InterpolationMode

from models import get_model


DIR = "/home/leeping/Clients/PMY/crowd_counting_test_dataset/"
WEIGHTS_PATH = "weights/clip_resnet50_word_448_8_4_fine_1.0_dmcount_aug/best_mae.pth"
VIDEO_OUTPUT_PATH = "results/pred.mp4"

# Model configs
dataset_name = "nwpu"
split = "val"
truncation = 4
reduction = 8
granularity = "fine"
anchor_points = "average"
model_name = "clip_resnet50"
input_size = 448
weight_count_loss = 1.0
count_loss = "dmcount"
prompt_type = "word"
augment = True

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Set parameters for CLIP-EBC evaluation', add_help=False)

    parser.add_argument('--threshold', default=0.5, type=float,
                        help="confidence threshold")
    parser.add_argument('--video_path', required=True,
                        help='path of video')
    parser.add_argument('--output_dir', default='',
                        help='path where to save')
    parser.add_argument('--weights_path', default=WEIGHTS_PATH,
                        help='path where the trained weights saved')
    parser.add_argument('--config_path', default="configs/reduction_8.json")

    return parser


def load_model(config_path, weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if truncation is None:  # regression, no truncation.
        bins, anchor_points = None, None
    else:
        with open(config_path, "r") as f:
            config = json.load(f)[str(truncation)][dataset_name]
        bins = config["bins"][granularity]
        # if anchor_points == "average" else config["anchor_points"][granularity]["middle"]
        anchor_points = config["anchor_points"][granularity]["average"]
        bins = [(float(b[0]), float(b[1])) for b in bins]
        anchor_points = [float(p) for p in anchor_points]

    model = get_model(
        backbone=model_name,
        input_size=input_size,
        reduction=reduction,
        bins=bins,
        anchor_points=anchor_points,
        prompt_type=prompt_type
    )

    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    return model, device


def inference(model, device, img_raw, threshold):

    input_w, input_h = img_raw.shape[1], img_raw.shape[0]

    # preprocess, following Viso repo steps
    to_tensor = ToTensor()
    list_processed = []
    image_org_tensor = to_tensor(img_raw)
    image_normal_tensor = normalize(image_org_tensor, mean=mean, std=std)
    image_normal_tensor = image_normal_tensor.unsqueeze(0)
    image_normal_tensor = image_normal_tensor.to(device)
    list_processed.append(image_normal_tensor)
    batch_tensor = torch.cat(list_processed, dim=0)

    # inference
    with torch.no_grad():
        pred_density = model(batch_tensor)  # [1, 1, 96, 160]

    output_h, output_w = pred_density.detach().cpu(
    ).shape[2], pred_density.detach().cpu().shape[3]
    pred = np.asarray(pred_density.detach().cpu().reshape(output_h, output_w))

    # filter by threshold
    pred[pred < threshold] = 0
    pred_min = np.min(pred)
    pred[pred < abs(pred_min)] = 0

    # draw heatmap
    heatmap = cv2.resize(pred, dsize=(int(input_w), int(
        input_h)), interpolation=cv2.INTER_CUBIC)
    # Draw red shade over detected people
    heatmap[heatmap < threshold] = 0
    heatmap[heatmap >= threshold] = 1

    show_img = img_raw.copy()
    mask = heatmap == 1
    show_img[mask, 2] = 255

    return show_img


def main(args):

    model, device = load_model(args.config_path, args.weights_path)

    vid_capture = cv2.VideoCapture(args.video_path)
    frame_width = int(vid_capture.get(3))
    frame_height = int(vid_capture.get(4))
    new_width = 1024  # frame_width // 128 * 128
    new_height = 512  # frame_height // 128 * 128
    vid_output = cv2.VideoWriter(
        VIDEO_OUTPUT_PATH, cv2.VideoWriter_fourcc(
            *"MP4V"), 30, (frame_width, frame_height)
    )

    start_time = time.time()
    num_frames = 0
    while vid_capture.isOpened():
        ret, frame = vid_capture.read()
        if ret == True:

            num_frames += 1
            img_raw = cv2.resize(frame, (new_width, new_height))
            img_to_draw = inference(model, device, img_raw, args.threshold)

            if cv2.waitKey(20) == ord("q"):
                break

            cv2.imshow("Frame", img_to_draw)
            vid_output.write(img_to_draw)
        else:
            break

    # Release the video capture object
    print(f"FPS: {(num_frames/(time.time()-start_time)):.2f}")
    vid_capture.release()
    vid_output.release()
    cv2.destroyAllWindows()
    # img_list = os.listdir(os.path.join(DIR, "images"))
    # results = [
    #     ["Image", "Ground Truth", "Prediction", "Absolute Error"]
    # ]

    # for img_filename in tqdm(img_list):
    #     img_filename_no_ext = img_filename.split(".")[0]
    #     img_path = os.path.join(DIR, "images", img_filename)
    #     json_path = os.path.join(
    #         DIR, "annotation", img_filename_no_ext + ".json")
    #     with open(json_path, "r") as file:
    #         annotations = json.load(file)

    #     pred_count = inference(img_path, model, device,
    #                            args.threshold, args.output_dir)

    #     results.append([img_filename, annotations["human_num"],
    #                    pred_count, abs(annotations["human_num"] - pred_count)])

    # with open("eval_results.csv", "w", newline="") as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerows(results)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        'CLIP-EBC evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
