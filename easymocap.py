import os

import cv2
from easy_ViTPose import VitInference

import queue
import threading
import argparse
import pickle
import numpy as np
from tqdm import tqdm
import torch


def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", '--scene', nargs="+", type=str,
                        help="Scene to be processed, be a list of PATH to scene base root")
    parser.add_argument("-v", '--view', nargs="+", type=str, help="View number")
    parser.add_argument("-o", '--output', type=str, help="Output dir, a relative path to the scene base root")
    parser.add_argument("--start", default=0, type=int, help="Start frame")
    parser.add_argument("--end", default=-1, type=int, help="End frame")
    parser.add_argument("--step", default=1, type=int, help="Step frame")
    parser.add_argument("--image", default=False, action="store_true", help="Process video instead of images")

    args = parser.parse_args()

    print("Args: ", args)
    print("output file structure: ")
    # check all scenes exits
    for scene in args.scene:
        assert os.path.exists(scene), f"Scene {scene} does not exist"
        # check output path exists
        if not os.path.exists(os.path.join(scene, args.output)):
            os.makedirs(os.path.join(scene, args.output))
        else:
            print(f"Output path {os.path.join(scene, args.output)} already exists")
        # like a tree
        # print(f"{scene}")
        # print(f"├── {args.output}")
        # check view video exists
        for view in args.view:
            if not os.path.exists(os.path.join(scene, "videos", f"{view}.mp4")):
                raise FileNotFoundError(f"View {view} does not exist")
            # print(f"│   ├── {view}")
            if not os.path.exists(os.path.join(scene, args.output, view)):
                os.makedirs(os.path.join(scene, args.output, view))
    return args


def read_images(image_base, frame_id_list, frame_queue):
    for frame_id in frame_id_list:
        img_path = os.path.join(image_base, f"{frame_id:06d}.jpg")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_queue.put((frame_id, img))

def read_videos(capture, frame_queue):
    while True:
        ret, frame = capture.read()
        frame_queue.put((ret, frame))
        if not ret:
            break
def main(args):
    assert torch.cuda.is_available(), "CUDA is not available!"
    # prepare model
    model_path = './models/vitpose-l-wholebody.pth'
    yolo_path = './models/yolov8l.pt'
    model = VitInference(model_path, yolo_path, model_name='l', yolo_size=640, is_video=True, device=None)

    for scene in args.scene:
        for view in args.view:
            # prepare queue for multi-threading image reading
            q_read = queue.Queue(maxsize=20)
            keypoints_dict = {}

            video_path = os.path.join(scene, "videos")
            capture = cv2.VideoCapture(os.path.join(video_path, f"{view}.mp4"))
            capture.set(cv2.CAP_PROP_POS_FRAMES, args.start - 1)

            # image_base = str(os.path.join(scene, "images", view))
            frame_id_list = list(
                range(args.start, args.end if args.end != -1 else int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), args.step))
            # threading.Thread(target=read_images, args=(image_base, frame_id_list, q_read)).start()


            threading.Thread(target=read_videos, args=(capture, q_read)).start()

            last_frame_id = frame_id_list[0]
            for frame_id in tqdm(frame_id_list, desc=f"Processing {scene}/{view}"):
                _, img = q_read.get()

                keypoints = model.inference(img)
                keypoints_dict[frame_id] = np.array([kpt[:23, [1, 0, 2]] for kpt in keypoints.values()])
                # save keypoints
                if len(keypoints_dict) == 1000:
                    with open(os.path.join(scene, args.output, view, f"{last_frame_id:06d}.pkl"), "wb") as f:
                        pickle.dump(keypoints_dict, f)
                    keypoints_dict = {}
                    last_frame_id = frame_id
            # save the rest
            if frame_id != last_frame_id:
                with open(os.path.join(scene, args.output, view, f"{frame_id:06d}.pkl"), "wb") as f:
                    pickle.dump(keypoints_dict, f)


            model.reset()
        # call model.reset() after each video


if __name__ == '__main__':
    args = prepare_args()
    main(args)
