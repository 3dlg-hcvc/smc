import numpy as np
from components.bounding_box import BoundingBox

def sampling_iou(bbox1: BoundingBox, bbox2: BoundingBox, num_samples: int = 10000) -> float:
    '''
    Calculate the IoU between two bounding boxes by sampling points in the boxes.

    Args:
        bbox1: BoundingBox, the first bounding box
        bbox2: BoundingBox, the second bounding box
        num_samples: int, the number of samples to sample in each bounding box
    
    Returns:
        iou: float, the IoU between the two bounding boxes
    '''

    # Sample points in the bounding boxes
    points1 = bbox1.sample_points(num_samples)
    points2 = bbox2.sample_points(num_samples)

    # Calculate the IoU
    points2_in_bbox1 = np.sum(bbox1.points_in_box(points2))
    points1_in_bbox2 = np.sum(bbox2.points_in_box(points1))

    # IoU weighted by volume
    intersect = (bbox1.volume * points1_in_bbox2 + bbox2.volume * points2_in_bbox1) / 2
    union = bbox1.volume * num_samples + bbox2.volume * num_samples - intersect
    iou = intersect / union

    return iou
