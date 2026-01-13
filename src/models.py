from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# Where things are in 3D Space
class Position(BaseModel):
    """Represents 3D Position (x,y,z)"""
    x: float = Field(..., description='x position (meters)')
    y: float = Field(..., description='y position (meters)')
    z: float = Field(..., description='z position (meters)')

# What the robot sees
class DetectedObject(BaseModel):
    """An object detected in the scene"""
    name: str = Field(..., description='Object identifier (eg. red block')
    object_type: str = Field(..., description='Category (eg. block, cup')
    position: Position
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

# The scene
class Scene(BaseModel):
    """Complete scene description from vision"""
    objects: List[DetectedObject] = Field(default_factory=list)
    description: Optional[str] = Field(None, description='Scene summary')

# Commands the robot executes
class RobotAction(BaseModel):
    """A single action for the robot to preform"""
    type: Literal['move to', 'grasp', 'release', 'look_at'] = Field(..., description='Type of action')
    target: str = Field(..., description='Object to act on')
    end_effector: str = Field(default='right hand', description='Which hand/gripper')
    position: Optional[Position] = Field(None, description='Target position if needed')
    parameters: dict = Field(default_factory=dict, description='Action parameters')

# The full sequence
class ActionPlan(BaseModel):
    """Complete plan with multiple actions"""
    actions: List[RobotAction]
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: Optional[str] = Field(None, description='Reasoning of the action')

# Input from user
class Command(BaseModel):
    """User's command"""
    text: str = Field(..., description='The command')
    image_path: Optional[str] = Field(None, description='Image path')

