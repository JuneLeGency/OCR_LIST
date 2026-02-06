#!/usr/bin/env python3
"""
Quick launcher for OCR processing and evaluation pipeline
"""

import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description='OCR Invoice Processing and Evaluation Pipeline'
    )
    parser.add_argument(
        '--process',
        action='store_true',
        help='Run OCR processing on all invoices'
    )
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Run accuracy evaluation (requires results)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run both processing and evaluation'
    )
    parser.add_argument(
        '--input-dir',
        default='./inputs',
        help='Input directory with invoice images (default: ./inputs)'
    )

    args = parser.parse_args()

    if args.all:
        args.process = True
        args.evaluate = True

    if not (args.process or args.evaluate):
        args.process = True  # Default action

    if args.process:
        print("="*80)
        print("STARTING OCR PROCESSING")
        print("="*80)
        try:
            from invoice_ocr_processor import InvoiceOCRProcessor
            processor = InvoiceOCRProcessor(input_dir=args.input_dir)
            processor.process_all_invoices()
            processor.generate_report()
            print("\n✓ OCR processing completed successfully")
        except Exception as e:
            print(f"\n✗ Error during processing: {e}")
            import traceback
            traceback.print_exc()
            return 1

    if args.evaluate:
        print("\n" + "="*80)
        print("STARTING ACCURACY EVALUATION")
        print("="*80)
        try:
            from evaluate_ocr_accuracy import OCRAccuracyEvaluator
            from pathlib import Path

            results_dir = Path("./ocr_results")
            results_files = sorted(results_dir.glob("results_*.json"), reverse=True)

            if not results_files:
                print("No results files found. Run processing first with --process flag.")
                return 1

            latest_results = results_files[0]
            ground_truth_file = Path("./ground_truth.json")

            evaluator = OCRAccuracyEvaluator(
                str(latest_results),
                str(ground_truth_file) if ground_truth_file.exists() else None
            )

            if evaluator.ground_truth:
                evaluator.evaluate_against_ground_truth()
            else:
                print("Ground truth file not found. Running consistency analysis instead.\n")

            evaluator.evaluate_model_consistency()
            print("\n✓ Accuracy evaluation completed successfully")
        except Exception as e:
            print(f"\n✗ Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
