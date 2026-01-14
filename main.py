import argparse
import json
import sys
import os
from pathlib import Path

from src.planner import ActionPlanner
from src.vision import VisionProcessor

def main():
    """ Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Generate robot action plans from commands',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  %(prog)s "pick up the red block" --scene scene1.jpg
  %(prog)s "move the apple to the right" --scene scene2.jpg --output plan.json
  %(prog)s "look at the yellow ball" --scene scene3.jpg --verbose

Available mock scenes:
  scene1.jpg - Table with red and blue blocks
  scene2.jpg - Kitchen counter with mug and apples  
  scene3.jpg - Stacked blocks and yellow ball
  (default)  - Simple scene with one red block
        """
    )

    # Required arguments
    parser.add_argument(
        'command',
        type=str,
        help='Command (eg. "pick up the red block")'
    )

    # Optional arguments
    parser.add_argument(
        '--scene',
        type=str,
        default=None,
        help='Scene name (eg. "scene1.jpg")'
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output JSON file path (default: print to stdout)"
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Groq API key (default: use GROQ_API_KEY environment variable)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.1-8b-instant",
        help="LLM model to use (default: llama-3.1-8b-instant)"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed information during processing"
    )

    parser.add_argument(
        "--list-scenes",
        action="store_true",
        help="List available mock scenes and exit"
    )

    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output (indented)"
    )

    args = parser.parse_args()

    if args.list_scenes:
        print('Available mock scenes:')
        vision = VisionProcessor(mock_mode=True)
        for scene_name in vision.list_available_scenes():
            print(f' - {scene_name}')
        return 0

    api_key = args.api_key or os.getenv('GROQ_API_KEY')
    if not api_key:
        print("ERROR: Groq API key required.", file=sys.stderr)
        print("Set GROQ_API_KEY environment variable or use --api-key", file=sys.stderr)
        return 1

    try:
        # print verbose info
        if args.verbose:
            print(f"Command: {args.command}", file=sys.stderr)
            print(f"Scene: {args.scene or '(default)'}", file=sys.stderr)
            print(f"Model: {args.model}", file=sys.stderr)
            print("Processing...", file=sys.stderr)

        planner = ActionPlanner(
            vision_mock_mode=True,
            llm_api_key=api_key,
            llm_model=args.model,
        )

        # Generate plan
        plan = planner.plan(args.command, args.scene)

        # convert to json
        if args.pretty:
            json.output = plan.model_dump_json(indent=2)
        else:
            json_output = plan.model_dump_json()

        if args.output:
            # write to file
            output_path =  Path(args.output)
            output_path.write_text(json_output)
            if args.verbose:
                print(f"\n✓ Plan saved to: {args.output}", file=sys.stderr)
                print(f"  Actions: {len(plan.actions)}", file=sys.stderr)
                print(f"  Confidence: {plan.confidence:.2f}", file=sys.stderr)

        else:
            print(json_output)

        if args.verbose:
            print("\n" + "=" * 50, file=sys.stderr)
            print("PLAN SUMMARY", file=sys.stderr)
            print("=" * 50, file=sys.stderr)
            print(f"Actions: {len(plan.actions)}", file=sys.stderr)
            print(f"Confidence: {plan.confidence:.2f}", file=sys.stderr)
            if plan.reasoning:
                print(f"Reasoning: {plan.reasoning}", file=sys.stderr)
            print("\nAction sequence:", file=sys.stderr)
            for i, action in enumerate(plan.actions, 1):
                print(f"  {i}. {action.type} {action.target}", file=sys.stderr)


        return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        if args.verbose:
           import traceback

           traceback.print_exc(file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())