import json
import os
import pathlib

import importlib_metadata
import numpy as np
import pandas as pd
import torch
#from IPython import get_ipython
#from neptunecontrib.api import log_table
from torchvision.models.detection.transform import GeneralizedRCNNTransform
#from torchvision.ops import box_convert, box_area

from metrics.bounding_box import BoundingBox
from metrics.enumerators import BBFormat, BBType
#from torchvision.ops import box_convert, box_area, nms
from model.roi_layers import nms

BoundingBoxforTestset = dict()

def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
    """
    Returns a list of files in a directory/path. Uses pathlib.
    """
    filenames = [file for file in path.glob(ext) if file.is_file()]
    assert len(filenames) > 0, f'No files found in path: {path}'
    return filenames


def read_json(path: pathlib.Path):
    with open(str(path), 'r') as fp:  # fp is the file pointer
        file = json.loads(s=fp.read())

    return file


def save_json(obj, path: pathlib.Path):
    with open(path, 'w') as fp:  # fp is the file pointer
        json.dump(obj=obj, fp=fp, indent=4, sort_keys=False)


def collate_double(batch):
    """
    collate function for the ObjectDetectionDataSet.
    Only used by the dataloader.
    """
    x = [sample['x'] for sample in batch]
    y = [sample['y'] for sample in batch]
    x_name = [sample['x_name'] for sample in batch]
    y_name = [sample['y_name'] for sample in batch]
    return x, y, x_name, y_name


def collate_single(batch):
    """
    collate function for the ObjectDetectionDataSetSingle.
    Only used by the dataloader.
    """
    x = [sample['x'] for sample in batch if sample != 'not an image']
    x_name = [sample['x_name'] for sample in batch if sample != 'not an image']
    return x, x_name


def color_mapping_func(labels, mapping):
    """Maps an label (integer or string) to a color"""
    color_list = [mapping[value] for value in labels]
    return color_list


def from_file_to_boundingbox(file_name: pathlib.Path, groundtruth: bool = True):
    """Returns a list of BoundingBox objects from groundtruth or prediction."""
    file = torch.load(file_name)
    labels = file['labels']
    boxes = file['boxes']
    scores = file['scores'] if not groundtruth else [None] * len(boxes)

    gt = BBType.GROUND_TRUTH if groundtruth else BBType.DETECTED

    return [BoundingBox(image_name=file_name.stem,
                        class_id=l,
                        coordinates=tuple(bb),
                        format=BBFormat.XYX2Y2,
                        bb_type=gt,
                        confidence=s) for bb, l, s in zip(boxes, labels, scores)]


# def from_dict_to_boundingbox(file: dict, name: str, groundtruth: bool = True):
#     """Returns list of BoundingBox objects from groundtruth or prediction."""
#     labels = file['labels']
#     boxes = file['boxes']
#     scores = np.array(file['scores'].cpu()) if not groundtruth else [None] * len(boxes)

#     gt = BBType.GROUND_TRUTH if groundtruth else BBType.DETECTED

#     return [BoundingBox(image_name=name,
#                         class_id=int(l),
#                         coordinates=tuple(bb),
#                         format=BBFormat.XYX2Y2,
#                         bb_type=gt,
#                         confidence=s) for bb, l, s in zip(boxes, labels, scores)]

def from_dict_to_boundingbox(file: dict, name: str, groundtruth: bool = True, testStep: bool = False):
    """Returns list of BoundingBox objects from groundtruth or prediction."""
    labels = file['labels']
    boxes = file['boxes']
    
    # print('boxes as list length: '+ str(len(boxes.tolist())))
    scores = np.array(file['scores'].cpu()) if not groundtruth else [None] * len(boxes)
    gt = BBType.GROUND_TRUTH if groundtruth else BBType.DETECTED
    
    if not groundtruth:
        indicesAfterNMS = nms(boxes, file['scores'], 0.6) #iou_threshold = 0.6 is adjustable
        indicesAfterNMS = indicesAfterNMS.tolist()

        boxes = [boxes[i] for i in indicesAfterNMS]
        labels = [labels[j] for j in indicesAfterNMS]
        scores = [scores[k] for k in indicesAfterNMS]

    if testStep:
        if name not in BoundingBoxforTestset:
            BoundingBoxforTestset[name] = dict()
            BoundingBoxforTestset[name]['generated_bb'] = []
            BoundingBoxforTestset[name]['groundtruth_bb'] = []
            BoundingBoxforTestset[name]['generated_label'] = []
            BoundingBoxforTestset[name]['groundtruth_label'] = []
            BoundingBoxforTestset[name]['scores'] = []
        if not groundtruth:
            BoundingBoxforTestset[name]['generated_bb'].append(boxes)
            BoundingBoxforTestset[name]['generated_label'].append(labels)
            BoundingBoxforTestset[name]['scores'].append(scores)
            # print('all generated boxes in test step: ')
            # print(boxes)
        elif groundtruth:
            BoundingBoxforTestset[name]['groundtruth_bb'].append(boxes.tolist())
            BoundingBoxforTestset[name]['groundtruth_label'].append(labels.tolist())
        print('Generated bb length: ' + str(len(BoundingBoxforTestset[name]['generated_bb'])))
        # print('\n')
        torch.save(BoundingBoxforTestset, '/Users/ashleykwon/Desktop/PyTorch-Object-Detection-Faster-RCNN-Tutorial-master/pytorch_faster_rcnn_tutorial/boundingBoxDict_nms_applied.json')
        # print('labels length in from dict to bounding box: '+ str(len(labels))) #added
    return [BoundingBox(image_name=name,
                        class_id=int(l),
                        coordinates=tuple(bb),
                        format=BBFormat.XYX2Y2,
                        bb_type=gt,
                        confidence=s) for bb, l, s in zip(boxes, labels, scores)]


def log_model_neptune(checkpoint_path: pathlib.Path,
                      save_directory: pathlib.Path,
                      name: str,
                      neptune_logger):
    """Saves the model to disk, uploads it to neptune and removes it again."""
    checkpoint = torch.load(checkpoint_path)
    model = checkpoint['hyper_parameters']['model']
    torch.save(model.state_dict(), save_directory / name)
    neptune_logger.experiment.set_property('checkpoint_name', checkpoint_path.name)
    neptune_logger.experiment.log_artifact(str(save_directory / name))
    if os.path.isfile(save_directory / name):
        os.remove(save_directory / name)


def log_checkpoint_neptune(checkpoint_path: pathlib.Path, neptune_logger):
    neptune_logger.experiment.set_property('checkpoint_name', checkpoint_path.name)
    neptune_logger.experiment.log_artifact(str(checkpoint_path))


from collections import defaultdict, deque
import time
import datetime
import torch.distributed as dist

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )

class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")
