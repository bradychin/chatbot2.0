import os
from typing import Optional

from src.models import Command, Scene, ActionPlan
from src.vision import VisionProcessor
from src.llm import LLMClient

class ActionPlanner:
    """
    Main planner that coordinates vision and LLM to generate robot action plans
    """

    def __init__(
            self,
            vision_mock_mode: bool = True,
            llm_provider: str = 'groq',
            llm_api_key: Optional[str] = None,
            llm_model: str = 'llama-3.1-8b-instant'
    ):
        """
        Initializes the action planner
        :param vision_mock_mode: if True, use mock vision. If False, use real CV
        :param llm_provider: LLM provider to use ("groq", future: "ollama", "openai")
        :param llm_api_key: API key for LLM provider
        :param llm_model: Model name to use for LLM
        """
        self.vision = VisionProcessor(mock_mode=vision_mock_mode)
        self.llm = LLMClient(
            provider=llm_provider,
            api_key=llm_api_key,
            model=llm_model
        )

    def plan(self, command_text: str, image_path: Optional[str] = None) -> ActionPlan:
        """
        Generate a robot action plan from a natural language command
        This is the main entry point.
        this will:
            1. Creates command object from text
            2. Processes image through vision system
            3. Generates plan using LLM

        :param command_text: Users command
        :param image_path: Path to scene image
        :return: ActionPlan with sequence of robot actions
        """

        command = Command(text=command_text, image_path=image_path)
        scene = self.vision.process(image_path)
        plan = self.llm.generate_plan(command, scene)

        return plan

    def plan_from_command(self, command: Command) -> ActionPlan:
        """
        Generate a plan from a Command object
        :param command: Command object with text
        :return: ActionPlan with robot actions
        """

        scene = self.vision.process(command.image_path)
        plan = self.llm.generate_plan(command, scene)

        return plan

    def plan_with_scene(self, command: Command, scene: Scene) -> ActionPlan:
        """
        Generate a plan when you already have the scene
        :param command: Command object
        :param scene: Pre-processed Scene object
        :return: ActionPlan with robot actions
        """
        return self.llm.generate_plan(command, scene)

def create_plan(
        command_text: str,
        image_path: Optional[str] = None,
        api_key: Optional[str] = None
) -> ActionPlan:
    """
    Convenience function to create a plan without instancing a planner
    :param command_text: Command
    :param image_path: Path to scene image
    :param api_key: Groq api key
    :return: ActionPlan with robot actions
    """

    planner = ActionPlanner(llm_api_key=api_key)
    return planner.plan(command_text, image_path)

























