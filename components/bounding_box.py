import numpy as np
import numpy.typing as npt

class BoundingBox():
    def __init__(self, 
                 centroid: npt.ArrayLike, 
                 half_size: npt.ArrayLike, 
                 coord_axes: npt.ArrayLike,
                 matrix_order: str = "F",
                 round_decimals: int = 5) -> None:
        '''
        Initialize a bounding box.

        Args:
            centroid: 1x3 vector, the centroid of the bounding box
            half_size: 1x3 vector, the half size of the bounding box
            coord_axes: 3x3 matrix, the basis of the coordinate system of the bounding box, each column is a unit axis
            matrix_order: string, the order for reshaping the coord_axes matrix into a 3x3 matrix
            round_decimals: int, the number of decimals to round the numbers to

        Returns:
            None
        '''

        # Initialize the properties
        self._round_decimals = round_decimals
        self._matrix_order = matrix_order
        self._centroid = None
        self._half_size = None
        self._coord_axes = None

        # Set the properties
        self.centroid = centroid
        self.half_size = half_size
        self.coord_axes = coord_axes

    # Main properties of the bounding box
    @property
    def centroid(self) -> np.ndarray:
        return self._centroid
    @property
    def half_size(self) -> np.ndarray:
        return self._half_size
    @property
    def coord_axes(self) -> np.ndarray:
        return self._coord_axes
    
    # Corresponding setters for rounding and type conversion
    @centroid.setter
    def centroid(self, centroid: npt.ArrayLike) -> None:
        self._centroid = np.array(centroid).round(self._round_decimals).astype(float)
    @half_size.setter
    def half_size(self, half_size: npt.ArrayLike) -> None:
        self._half_size = np.array(half_size).round(self._round_decimals).astype(float)
    @coord_axes.setter
    def coord_axes(self, coord_axes: npt.ArrayLike) -> None:
        self._coord_axes = np.array(coord_axes).round(self._round_decimals).astype(float).reshape(3, 3, order=self._matrix_order)
    
    @property
    def full_size(self) -> np.ndarray:
        '''
        Return the full size of the bounding box.

        Returns:
            full_size: 1x3 vector, the full size of the bounding box
        '''

        return self.half_size * 2

    @property
    def volume(self) -> float:
        '''
        Return the volume of the bounding box.

        Returns:
            volume: float, the volume of the bounding box
        '''

        return np.prod(self.full_size)
    
    @property
    def matrix(self) -> np.ndarray:
        '''
        Return the 4x4 transformation matrix of the bounding box.

        Returns:
            matrix: 4x4 matrix, the transformation matrix of the bounding box
        '''

        matrix = np.eye(4)
        matrix[:3, :3] = self.coord_axes @ np.diag(self.half_size)
        matrix[:3, 3] = self.centroid

        return matrix
    
    @property
    def no_scale_matrix(self) -> np.ndarray:
        '''
        Return the 4x4 transformation matrix of the bounding box without scaling.

        Returns:
            matrix: 4x4 matrix, the transformation matrix of the bounding box without scaling
        '''

        matrix = np.eye(4)
        matrix[:3, :3] = self.coord_axes
        matrix[:3, 3] = self.centroid

        return matrix
    
    def sample_points(self, num_samples: int = 10000) -> np.ndarray:
        '''
        Sample points in the bounding box.

        Args:
            num_samples: int, the number of samples to sample in the bounding box
        
        Returns:
            points: num_samplesx3 matrix, the sampled points in the bounding box
        '''

        points = np.random.rand(num_samples, 3) * 2 - 1     # sample points in the [-1, 1]^3 cube centered at the origin
        points = points * self.half_size                    # scale the points to the bounding box's size
        points = (self.coord_axes @ points.T).T             # rotate the points to align with the bounding box's frame
        points = points + self.centroid                     # translate the points to the bounding box's position

        return points

    def points_in_box(self, points: npt.NDArray) -> list[bool]:
        '''
        Check if the points are in the bounding box.

        Args:
            points: nx3 matrix, the points to check
        
        Returns:
            in_box: list of booleans, whether the points are in the bounding box
        '''

        points_local = points - self.centroid
        points_local = (self.coord_axes.T @ points_local.T).T
        in_box = np.all(np.abs(points_local) <= self.half_size, axis=1)

        return in_box
