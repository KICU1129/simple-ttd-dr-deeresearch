"""
This is the main entry point for running the TTD-DR agent.
"""
import argparse
import logging
import os
import time
from datetime import datetime
from .controller import TTD_DR_Controller


def setup_logging():
    """Configures the root logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            # Optional: Add a FileHandler to log to a file
            # logging.FileHandler("ttd_dr_run.log") 
        ]
    )

def main():
    """
    Main function to run the TTD-DR controller from the command line.
    """
    
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Run the Test-Time Diffusion Deep Researcher.")
    parser.add_argument("query", type=str, help="The research query to process.")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="The maximum number of refinement iterations."
    )
    args = parser.parse_args()

    logger.info(f"Starting TTD-DR with query: '{args.query}'")
    start_time = time.time()
    
    controller = TTD_DR_Controller(max_iterations=args.max_iterations)
    final_state = controller.run(args.query)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # --- Save Final Report ---
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(output_dir, f"report_{timestamp}.md")
    
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(final_state.final_report)
        logger.info(f"Final report successfully saved to {file_path}")
    except IOError as e:
        logger.error(f"Failed to save final report: {e}")


    # --- Print Final Summary ---
    logger.info("\n\n" + "="*20 + " FINAL REPORT " + "="*20)
    logger.info(final_state.final_report)
    logger.info("="*54)

    # --- Print Execution Summary ---
    logger.info("\n" + "="*20 + " Execution Summary " + "="*20)
    logger.info(f"Total Execution Time: {elapsed_time:.2f} seconds")
    logger.info(f"Total Tokens Used (Approximate): {final_state.total_tokens}")
    
    unique_citations = sorted(list(set(final_state.citations)))
    logger.info(f"Total Citations Found: {len(unique_citations)}")
    if unique_citations:
        logger.info("Citations:")
        for url in unique_citations:
            logger.info(f"- {url}")
    logger.info("="*57)


if __name__ == "__main__":
    main()
