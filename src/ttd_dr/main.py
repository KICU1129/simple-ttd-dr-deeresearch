"""
This is the main entry point for running the TTD-DR agent.
"""
import argparse
from .controller import TTD_DR_Controller

def main():
    """
    Main function to run the TTD-DR controller from the command line.
    """
    parser = argparse.ArgumentParser(description="Run the Test-Time Diffusion Deep Researcher.")
    parser.add_argument("query", type=str, help="The research query to process.")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="The maximum number of refinement iterations."
    )
    args = parser.parse_args()

    print(f"Starting TTD-DR with query: '{args.query}'")
    
    controller = TTD_DR_Controller(max_iterations=args.max_iterations)
    final_report = controller.run(args.query)

    print("\n\n" + "="*20 + " FINAL REPORT " + "="*20)
    print(final_report)
    print("="*54)

if __name__ == "__main__":
    main()
