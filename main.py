import sys
import json
from pathlib import Path

from src.planner import ActionPlanner
from src.vision import VisionProcessor
from src.config import groq_api_key, mock_mode

def print_banner():
    """Print welcome banner"""
    print("=" * 60)
    print("  ROBOT COMMAND SYSTEM")
    print("=" * 60)
    print()

def print_plan_summary(plan):
    """Print summary of the plan"""
    print("\n" + "-" * 60)
    print("ACTIONS")
    print("-" * 60)
    print(f"Confidence: {plan.confidence:.2f}")
    if plan.reasoning:
        print(f"Reasoning: {plan.reasoning}")

    print(f"\nAction sequence ({len(plan.actions)} actions):")
    for i, action in enumerate(plan.actions, 1):
        pos_str = ""
        if action.position:
            pos_str = f" at ({action.position.x:.2f}, {action.position.y:.2f}, {action.position.z:.2f})"
        print(f"  {i}. {action.type.upper()}: {action.target}{pos_str}")
        print(f"     Using: {action.end_effector}")
    print("-" * 60)

def main():
    """Main function"""
    print_banner()

    api_key = groq_api_key
    if not api_key:
        print('ERROR: API key not provided')
        return 1

    try:
        vision = VisionProcessor(mock_mode=mock_mode)
        planner = ActionPlanner(
            vision_mock_mode=mock_mode,
            llm_api_key=api_key,
            llm_model='llama-3.1-8b-instant'
        )

        print("System initialized successfully!")

        # List avialable scenes
        scenes = vision.list_available_scenes()
        print("Available scenes:")
        for scene in scenes:
            print(f"  {scene}")

        # Have user pick scene
        print()
        choice = input('Pick a scene (number or name): ').strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(scenes):
                selected_scene = scenes[idx]
            else:
                selected_scene = 'default'
        else:
            selected_scene = choice if choice else 'default'

        while True:
            # List items in scene
            scene = vision.process(selected_scene)
            print(f'\nScene: {scene.description}')
            print('Objects in scene:')
            for obj in scene.objects:
                print(f' - {obj.name} ({obj.object_type}) at ({obj.position.x:.2f}, {obj.position.y:.2f}, {obj.position.z:.2f})")')

            print('\nType "quit" at any time to exit\n')

            print()
            command = input('You: ').strip()

            if not command:
                continue

            if command.lower() in ['quit', 'exit', 'q']:
                print('\nGoodbye!')
                break

                # Generate plan for the command
            try:
                print("\nProcessing...")
                plan = planner.plan(command, selected_scene)
                print_plan_summary(plan)
                print()
                command2 = input('Would you like to give another command? (y/n): ')
                if command2.lower() in ['y', 'yes']:
                    continue
                else:
                    print("\nGoodbye!")
                    break


            except Exception as e:
                print(f"\n✗ Error: {e}")
                print("Please try again with a different command.")

        return 0

    except Exception as e:
        print(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())