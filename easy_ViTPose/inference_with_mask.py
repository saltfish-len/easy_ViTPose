import abc
import os
from typing import Optional
import typing

import cv2
import numpy as np
import torch

from .configs.ViTPose_common import data_cfg
from .sort import Sort
from .vit_models.model import ViTPose
from .vit_utils.inference import draw_bboxes, pad_image
from .vit_utils.top_down_eval import keypoints_from_heatmaps
from .vit_utils.util import dyn_model_import, infer_dataset_by_path
from .vit_utils.visualization import draw_points_and_skeleton, joints_dict

try:
    import torch_tensorrt
except ModuleNotFoundError:
    pass

try:
    import onnxruntime
except ModuleNotFoundError:
    pass

__all__ = ['VitInferenceWithMask']
np.bool = np.bool_
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

DETC_TO_YOLO_YOLOC = {
    'human': [0],
    'cat': [15],
    'dog': [16],
    'horse': [17],
    'sheep': [18],
    'cow': [19],
    'elephant': [20],
    'bear': [21],
    'zebra': [22],
    'giraffe': [23],
    'animals': [15, 16, 17, 18, 19, 20, 21, 22, 23]
}


class VitInferenceWithMask:
    """
    Class for performing inference using ViTPose models with YOLOv8 human detection and SORT tracking.

    Args:
        model (str): Path to the ViT model file (.pth, .onnx, .engine).
        yolo (str): Path of the YOLOv8 model to load.
        model_name (str, optional): Name of the ViT model architecture to use.
                                    Valid values are 's', 'b', 'l', 'h'.
                                    Defaults to None, is necessary when using .pth checkpoints.
        det_class (str, optional): the detection class. if None it is inferred by the dataset.
                                   valid values are 'human', 'cat', 'dog', 'horse', 'sheep',
                                                    'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                                                    'animals' (which is all previous but human)
        dataset (str, optional): Name of the dataset. If None it's extracted from the file name.
                                 Valid values are 'coco', 'coco_25', 'wholebody', 'mpii',
                                                  'ap10k', 'apt36k', 'aic'
        yolo_size (int, optional): Size of the input image for YOLOv8 model. Defaults to 320.
        device (str, optional): Device to use for inference. Defaults to 'cuda' if available, else 'cpu'.
        is_video (bool, optional): Flag indicating if the input is video. Defaults to False.
        single_pose (bool, optional): Flag indicating if the video (on images this flag has no effect)
                                      will contain a single pose.
                                      In this case the SORT tracker is not used (increasing performance)
                                      but people id tracking
                                      won't be consistent among frames.
        yolo_step (int, optional): The tracker can be used to predict the bboxes instead of yolo for performance,
                                   this flag specifies how often yolo is applied (e.g. 1 applies yolo every frame).
                                   This does not have any effect when is_video is False.
    """

    def __init__(self, model: str,
                 mask_path: str,
                 model_name: Optional[str] = None,
                 dataset: Optional[str] = None,
                 device: Optional[str] = None,
                 is_video: Optional[bool] = False,
                 single_pose: Optional[bool] = False,
                 num_people: Optional[int] = 1):
        assert os.path.isfile(model), f'The model file {model} does not exist'
        assert os.path.exists(mask_path), f'The mask folder {mask_path} does not exist'

        # Device priority is cuda / mps / cpu
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.device = device
        self.is_video = is_video
        self.single_pose = single_pose
        self.num_people = num_people
        self.mask_path = mask_path
        self.reset()

        # State saving during inference
        self.save_state = True  # Can be disabled manually
        self._img = None
        self._keypoints = None
        self.frame_counter = 0

        # Use extension to decide which kind of model has been loaded
        use_onnx = model.endswith('.onnx')
        use_trt = model.endswith('.engine')

        # Extract dataset name
        if dataset is None:
            dataset = infer_dataset_by_path(model)

        assert dataset in ['mpii', 'coco', 'coco_25', 'wholebody', 'aic', 'ap10k', 'apt36k'], \
            'The specified dataset is not valid'

        # Dataset can now be set for visualization
        self.dataset = dataset

        assert model_name in [None, 's', 'b', 'l', 'h'], \
            f'The model name {model_name} is not valid'

        # onnx / trt models do not require model_cfg specification
        if model_name is None:
            assert use_onnx or use_trt, \
                'Specify the model_name if not using onnx / trt'
        else:
            # Dynamically import the model class
            model_cfg = dyn_model_import(self.dataset, model_name)

        self.target_size = data_cfg['image_size']
        if use_onnx:
            self._ort_session = onnxruntime.InferenceSession(model,
                                                             providers=['CUDAExecutionProvider',
                                                                        'CPUExecutionProvider'])
            inf_fn = self._inference_onnx
        else:
            self._vit_pose = ViTPose(model_cfg)
            self._vit_pose.eval()

            if use_trt:
                self._vit_pose = torch.jit.load(model)
            else:
                ckpt = torch.load(model, map_location='cpu')
                if 'state_dict' in ckpt:
                    self._vit_pose.load_state_dict(ckpt['state_dict'])
                else:
                    self._vit_pose.load_state_dict(ckpt)
                self._vit_pose.to(torch.device(device))

            inf_fn = self._inference_torch

        # Override _inference abstract with selected engine
        self._inference = inf_fn  # type: ignore

    def reset(self):
        """
        Reset the inference class to be ready for a new video.
        This will reset the internal counter of frames, on videos
        this is necessary to reset the tracker.
        """
        self.frame_counter = 0
    def udpate_mask_path(self, mask_path):
        self.mask_path = mask_path
    @classmethod
    def postprocess(cls, heatmaps, org_w, org_h):
        """
        Postprocess the heatmaps to obtain keypoints and their probabilities.

        Args:
            heatmaps (ndarray): Heatmap predictions from the model.
            org_w (int): Original width of the image.
            org_h (int): Original height of the image.

        Returns:
            ndarray: Processed keypoints with probabilities.
        """
        points, prob = keypoints_from_heatmaps(heatmaps=heatmaps,
                                               center=np.array([[org_w // 2,
                                                                 org_h // 2]]),
                                               scale=np.array([[org_w, org_h]]),
                                               unbiased=True, use_udp=True)
        return np.concatenate([points[:, :, ::-1], prob], axis=2)

    @abc.abstractmethod
    def _inference(self, img: np.ndarray) -> np.ndarray:
        """
        Abstract method for performing inference on an image.
        It is overloaded by each inference engine.

        Args:
            img (ndarray): Input image for inference.

        Returns:
            ndarray: Inference results.
        """
        raise NotImplementedError

    def get_mask(self):
        """
        输入一个图片的路径，返回这个视角对应的mask，list[ndarray]，根据self.num_people来确定人数
        masks[i]是第i个人的mask
        mask_path: [scene_root]/new_masks/[view_id]/[mask_id]/[frame_id].jpg
        :param mask_path: 图片的路径, [scene_root]/images/[view_id]/[frame_id].jpg
        :return: masks: list[ndarray]
        """

        masks = []
        for i in range(self.num_people):
            mask_path = os.path.join(self.mask_path, f"{i}/{self.frame_counter}.jpg")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            masks.append(mask)
        return masks

    @staticmethod
    def mask2bbox(mask):
        """
        将mask转换成bbox
        :param mask:
        :return: xyxy 形式的bbox， 分别是x_min, y_min, x_max, y_max
        """
        xyhw = cv2.boundingRect(mask)
        return np.array([xyhw[0], xyhw[1], xyhw[0] + xyhw[2], xyhw[1] + xyhw[3]])

    def get_mask_bbox(self):
        masks = self.get_mask()
        return [self.mask2bbox(mask) for mask in masks]

    def inference(self, img: np.ndarray) -> dict[typing.Any, typing.Any]:
        """
        Perform inference on the input image.

        Args:
            img (ndarray): Input image for inference in RGB format.

        Returns:
            dict[typing.Any, typing.Any]: Inference results.
        """

        # First use YOLOv8 for detection
        res_pd = np.empty((0, 5))
        results = None

        res_pd = np.array(self.get_mask_bbox())
        res_pd = np.concatenate([res_pd, np.ones((res_pd.shape[0], 1))], axis=1)
        self.frame_counter += 1

        frame_keypoints = {}
        ids = None

        # Prepare boxes for inference
        bboxes = res_pd[:, :4].round().astype(int)
        scores = res_pd[:, 4].tolist()
        pad_bbox = 10

        if ids is None:
            ids = range(len(bboxes))

        for bbox, id in zip(bboxes, ids):
            # TODO: Slightly bigger bbox
            bbox[[0, 2]] = np.clip(bbox[[0, 2]] + [-pad_bbox, pad_bbox], 0, img.shape[1])
            bbox[[1, 3]] = np.clip(bbox[[1, 3]] + [-pad_bbox, pad_bbox], 0, img.shape[0])

            # Crop image and pad to 3/4 aspect ratio
            img_inf = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            img_inf, (left_pad, top_pad) = pad_image(img_inf, 3 / 4)

            keypoints = self._inference(img_inf)[0]
            # Transform keypoints to original image
            keypoints[:, :2] += bbox[:2][::-1] - [top_pad, left_pad]
            frame_keypoints[id] = keypoints

        return frame_keypoints

    def draw(self, show_yolo=True, show_raw_yolo=False, confidence_threshold=0.5):
        """
        Draw keypoints and bounding boxes on the image.

        Args:
            show_yolo (bool, optional): Whether to show YOLOv8 bounding boxes. Default is True.
            show_raw_yolo (bool, optional): Whether to show raw YOLOv8 bounding boxes. Default is False.

        Returns:
            ndarray: Image with keypoints and bounding boxes drawn.
        """
        img = self._img.copy()

        img = np.array(img)[..., ::-1]  # RGB to BGR for cv2 modules
        for idx, k in self._keypoints.items():
            img = draw_points_and_skeleton(img.copy(), k,
                                           joints_dict()[self.dataset]['skeleton'],
                                           person_index=idx,
                                           points_color_palette='gist_rainbow',
                                           skeleton_color_palette='jet',
                                           points_palette_samples=10,
                                           confidence_threshold=confidence_threshold)
        return img[..., ::-1]  # Return RGB as original

    def pre_img(self, img):
        org_h, org_w = img.shape[:2]
        img_input = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR) / 255
        img_input = ((img_input - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
        return img_input, org_h, org_w

    @torch.no_grad()
    def _inference_torch(self, img: np.ndarray) -> np.ndarray:
        # Prepare input data
        img_input, org_h, org_w = self.pre_img(img)
        img_input = torch.from_numpy(img_input).to(torch.device(self.device))

        # Feed to model
        heatmaps = self._vit_pose(img_input).detach().cpu().numpy()
        return self.postprocess(heatmaps, org_w, org_h)

    def _inference_onnx(self, img: np.ndarray) -> np.ndarray:
        # Prepare input data
        img_input, org_h, org_w = self.pre_img(img)

        # Feed to model
        ort_inputs = {self._ort_session.get_inputs()[0].name: img_input}
        heatmaps = self._ort_session.run(None, ort_inputs)[0]
        return self.postprocess(heatmaps, org_w, org_h)
