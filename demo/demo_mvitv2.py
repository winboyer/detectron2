# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import torch
import tqdm

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
import detectron2.data.transforms as T
from detectron2.structures import BitMasks, Boxes, BoxMode, Keypoints, PolygonMasks, RotatedBoxes

# constants
WINDOW_NAME = "COCO detections"


class model_inference:
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        
#         print('cfg.dataloader.test========', cfg.dataloader.test)
        self.model = instantiate(cfg.model)
        self.model.to(cfg.train.device)
        self.model = create_ddp_model(self.model)
        DetectionCheckpointer(self.model).load(cfg.MODEL.WEIGHTS)
        self.model.eval()
        
        self.aug = T.ResizeShortestEdge(short_edge_length=800, max_size=1333)
        image_format="BGR"
        self.input_format = image_format
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def run_on_image(self, image):         
        with torch.no_grad():
            if self.input_format == "RGB":
                image = image[:, :, ::-1]
            
            height, width = image.shape[:2]
            image = self.aug.get_transform(image).apply_image(image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            image.to(self.model.device)
            
            inputs = {"image": image, "height": height, "width": width}
            
            predictions = self.model([inputs])[0]
            return predictions
        
            
def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

    
def setup_cfg(args):
    
    # load config from file and command-line arguments
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)
    
    return cfg
    

def _convert_boxes(boxes):
    """
    Convert different format of boxes to an NxB array, where B = 4 or 5 is the box dimension.
    """
    if isinstance(boxes, Boxes) or isinstance(boxes, RotatedBoxes):
#         print('isinstance(boxes, Boxes)=========', isinstance(boxes, Boxes))
#         print('isinstance(boxes, RotatedBoxes)=========', isinstance(boxes, RotatedBoxes))
        return boxes.tensor.detach().numpy()
    else:
        return np.asarray(boxes)
    
    
if __name__ == "__main__":

    args = get_parser().parse_args()
#     args = default_argument_parser().parse_args()
#     print('args======', args)
    cfg = setup_cfg(args)
#     print('cfg========', cfg)
#     print('cfg.model========', cfg.model)

    print('cfg.MODEL.WEIGHTS ======', cfg.MODEL.WEIGHTS)
#     print('args.input=========', args.input)
    
    demo = model_inference(cfg)
    cpu_device = torch.device("cpu")
    if args.input:
        print('args.input[0]========', args.input[0])
        if '.jp' in args.input[0]:
            img = read_image(args.input[0], format="BGR")
            outputs = demo.run_on_image(img)
#             print('outputs============', outputs)
            if "instances" in outputs:
                instances = outputs["instances"].to(cpu_device)
#                 print('instances==========', instances)
                boxes = instances.pred_boxes if instances.has("pred_boxes") else None
                boxes = _convert_boxes(boxes)

                scores = instances.scores if instances.has("scores") else None
                classes = instances.pred_classes.tolist() if instances.has("pred_classes") else None
#                 labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))

                print(len(boxes))
                for box in boxes:
                    x0, y0, x1, y1 = box
#                     print('box===========', box, x0, y0, x1, y1)
                    draw_img = img.astype(np.uint8)
                    draw_img = cv2.rectangle(draw_img, (int(x0),int(y0)), (int(x1),int(y1)), (0,255,0), 2)

                cv2.imwrite('test.jpg', draw_img)
                print('scores=========', scores)
                print('classes=========', classes)
        
        else: 
            filenames = os.listdir(args.input[0])
            output = './results'
            if not os.path.exists(output):
                os.makedirs(output)
            for filename in filenames:
                if '.jp' not in filename:
                    continue
                path = os.path.join(args.input[0], filename)
                print('filename, path==========', filename, path)
                img = read_image(path, format="BGR")
                outputs = demo.run_on_image(img)
#                     print('outputs============', outputs)
                if "instances" in outputs:
                    instances = outputs["instances"].to(cpu_device)
                    boxes = instances.pred_boxes if instances.has("pred_boxes") else None
                    boxes = _convert_boxes(boxes)

                    scores = instances.scores if instances.has("scores") else None
                    classes = instances.pred_classes.tolist() if instances.has("pred_classes") else None
                    
                    draw_img = img.astype(np.uint8)
#                     print(len(boxes))
                    for box in boxes:
                        x0, y0, x1, y1 = box
#                         print('box===========', box, x0, y0, x1, y1)
                        draw_img = cv2.rectangle(draw_img, (int(x0),int(y0)), (int(x1),int(y1)), (0,255,0), 2)
                    savepath = os.path.join(output, filename)
                    cv2.imwrite(savepath, draw_img)
                
                
                