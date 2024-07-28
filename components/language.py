import numpy as np
from components.bounding_box import BoundingBox
from components.obj import Obj

def create(label: str, 
           half_size: list[float]) -> Obj:
    '''
    Create an object with a bounding box with the given label and half size, centered at the origin.

    Args:
        label: string, the label of the object
        half_size: 1x3 vector, the half size of the bounding box
    
    Returns:
        obj: Obj, the object
    '''
    
    bounding_box = BoundingBox([0, 0, 0], half_size, [1, 0, 0, 0, 1, 0, 0, 0, 1])
    obj = Obj(label, bounding_box, None, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    return obj

def move(obj: Obj, x: float, y: float, z: float) -> None:
    '''
    Move an object's bounding box to a new position.

    Args:
        obj: Obj, the object
        x: float, the x coordinate of the new position
        y: float, the y coordinate of the new position
        z: float, the z coordinate of the new position
    
    Returns:
        None
    '''

    translation = np.array([x, y, z]).astype(float)
    obj.bounding_box.centroid += translation

def rotate(obj: Obj, axis: str, angle: float) -> None:
    '''
    Rotate an object's bounding box's coordinate system around an axis in the box's coordinate system.

    Args:
        obj: Obj, the object
        axis: string, the axis of rotation ("x", "y", or "z")
        angle: float, the angle of rotation in degrees
    
    Returns:
        None
    '''

    # Get the axis of rotation
    match axis:
        case "x":
            axis_of_rotation = obj.bounding_box.coord_axes[:, 0]
        case "y":
            axis_of_rotation = obj.bounding_box.coord_axes[:, 1]
        case "z":
            axis_of_rotation = obj.bounding_box.coord_axes[:, 2]
        case _:
            raise ValueError("Invalid axis")
    
    # Convert the angle to radians
    angle = np.radians(angle)

    # Create a rotation matrix by the Rodrigues' rotation formula
    kx, ky, kz = axis_of_rotation
    k_matrix = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
    rotation_matrix = np.eye(3) + np.sin(angle) * k_matrix + (1 - np.cos(angle)) * k_matrix @ k_matrix
    
    obj.bounding_box.coord_axes = rotation_matrix @ obj.bounding_box.coord_axes
