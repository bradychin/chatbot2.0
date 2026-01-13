from typing import Optional
from src.models import Scene, DetectedObject, Position

MOCK_SCENES = {
    "scene1": Scene(
        objects=[
            DetectedObject(
                name='red_block',
                object_type='block',
                position=Position(x=0.5, y=0.2, z=0.0),
                confidence=0.95
            ),
            DetectedObject(
                name='blue_block',
                object_type='block',
                position=Position(x=0.3, y=0.1, z=0.0),
                confidence=0.92
            )
        ],
        description='Table with red and blue blocks'
    ),

    'scene2': Scene(
        objects=[
            DetectedObject(
                name='coffee_mug',
                object_type='cup',
                position=Position(x=0.2, y=0.3, z=0.0),
                confidence=0.91
            ),
            DetectedObject(
                name='red_apple',
                object_type='fruit',
                position=Position(x=0.0, y=0.2, z=0.0),
                confidence=0.94
            ),
            DetectedObject(
                name='green_apple',
                object_type='fruit',
                position=Position(x=0.1, y=0.25, z=0.0),
                confidence=0.91
            )
        ],
        description='Kitchen counter with mug and apples'
    ),

    'scene3': Scene(
        objects=[
            DetectedObject(
                name='blue_block_base',
                object_type='block',
                position=Position(x=0.4, y=0.15, z=0.0),
                confidence=0.96
            ),
            DetectedObject(
                name="red_block_stacked",
                object_type="block",
                position=Position(x=0.4, y=0.15, z=0.05),
                confidence=0.88
            ),
            DetectedObject(
                name="yellow_ball",
                object_type="ball",
                position=Position(x=-0.2, y=0.3, z=0.0),
                confidence=0.93
            )
        ],
        description="Table with stacked blocks and a yellow ball"
    ),

    "default": Scene(
        objects=[
            DetectedObject(
                name="red_block",
                object_type="block",
                position=Position(x=0.3, y=0.2, z=0.0),
                confidence=0.90
            )
        ],
        description="Simple scene with one red block"
    )
}

class VisionProcessor:
    """
    Processes images to detect objects and return scene descriptions
    """
    def __init__(self, mock_mode: bool = True):
        """
        Initializes vision processor
        :param mock_mode: If True, returns predefined scenes. If false, use computer vision
        """
        self.mock_mode = mock_mode

    def process(self, image_path: Optional[str] = None) -> Scene:
        """
        Processes images to detect objects and return scene descriptions
        :param image_path: Path to image
        :return: Scene description
        """

        if self.mock_mode:
            return self._get_mock_scene(image_path)
        else:
            raise NotImplementedError('Real computer vision will be implemented later...')

    def _get_mock_scene(self, image_path: Optional[str]) -> Scene:
        """
        Return predefined mock scene based on image path.
        :param image_path: Path to image file
        :return: Scene object
        """
        if image_path is None:
            return MOCK_SCENES['default']

        filename = image_path.split('/')[-1]
        scene_name = filename.split('.')[0]

        return MOCK_SCENES.get(scene_name, MOCK_SCENES['default'])

    def list_available_scenes(self) -> list[str]:
        """
        Get list of available scenes
        :return: List of available scenes
        """
        return list(MOCK_SCENES.keys())

def get_scene(image_path: Optional[str] = None) -> Scene:
    """
    Get scene without creating a processor
    :param image_path: Path to image file
    :return: Scene object with detected objects
    """
    processor = VisionProcessor(mock_mode=True)
    return processor.process(image_path)




















