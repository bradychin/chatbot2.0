import sys
import json
from pathlib import Path

from src.planner import ActionPlanner
from src.vision import VisionProcessor
from src.config import groq_api_key, mock_mode

def print_banner():
    """Print welcome banner"""
    print("=" * 60)
    print("  ROBOT COMMAND SYSTEM - Interactive Mode")
    print("=" * 60)
    print()

def print_available_scenes(vision: VisionProcessor):
    """Print available scenes"""
    print('\nAvailable scenes:')
    scenes = vision.list_available_scenes()
    for scene in scenes:
        print(f'    - {scene}')
    print()

def print_plan_summary(plan):
    """Print summary of the plan"""
    print("\n" + "-" * 60)
    print("GENERATED PLAN")
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

def get_input(prompt: str, default: str = None) -> str:
    """Get input from user"""
    if default:
        user_input = input(f'{prompt} [{default}]: ').strip()
        return user_input if user_input else default
    else:
        return input(f'{prompt}: ').strip()

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
        print("\nCommands:")
        print("  • Type a robot command (e.g., 'pick up the red block')")
        print("  • Type 'scenes' to see available scenes")
        print("  • Type 'help' for more information")
        print("  • Type 'quit' or 'exit' to quit")
        print()

        current_scene = None

        while True:
            print()
            command = input('You: ').strip()

            if not command:
                continue

            if command.lower() in ['quit', 'exit', 'q']:
                print('\nGoodbye!')
                break

            if command.lower() in ['help', 'h', '?']:
                print("\nHow to use:")
                print("  1. Optionally specify a scene (or use default)")
                print("  2. Give a natural language command")
                print("  3. Review the generated action plan")
                print("\nExample commands:")
                print("  • pick up the red block")
                print("  • move the apple to the left")
                print("  • look at the yellow ball")
                continue

            if command.lower() in ['scenes', 'list', 'ls']:
                print_available_scenes(vision)
                continue

            if command.lower().startswith('scene '):
                # Allow user to change scene
                scene_name = command[6:].strip()
                current_scene = scene_name if scene_name else None
                print(f"Scene set to: {current_scene or 'default'}")
                continue

                # Generate plan for the command
            try:
                print("\nProcessing...")
                plan = planner.plan(command, current_scene)

                # Display results
                print_plan_summary(plan)

                # Ask if user wants to save
                save = get_input("\nSave to file? (y/n)", "n")
                if save.lower() in ['y', 'yes']:
                    filename = get_input("Filename", "plan.json")
                    output_path = Path(filename)
                    json_output = plan.model_dump_json(indent=2)
                    output_path.write_text(json_output)
                    print(f"✓ Saved to {filename}")

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