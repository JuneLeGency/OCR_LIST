#!/usr/bin/env python3
"""
OCR Accuracy Evaluation Script
Evaluates accuracy of different OCR models by comparing extracted amounts
against ground truth (manually verified amounts).
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import statistics


@dataclass
class AccuracyMetrics:
    """Store accuracy metrics for a model"""
    model_name: str
    total_invoices: int = 0
    correct_amounts: int = 0
    incorrect_amounts: int = 0
    extraction_failed: int = 0
    accuracy_percentage: float = 0.0
    average_error: float = 0.0
    max_error: float = 0.0
    errors: List[float] = field(default_factory=list)

    def calculate(self):
        if self.total_invoices > 0:
            self.accuracy_percentage = (self.correct_amounts / self.total_invoices) * 100
            if self.errors:
                self.average_error = statistics.mean([abs(e) for e in self.errors])
                self.max_error = max([abs(e) for e in self.errors])

    def to_dict(self):
        return {
            'model': self.model_name,
            'total': self.total_invoices,
            'correct': self.correct_amounts,
            'incorrect': self.incorrect_amounts,
            'extraction_failed': self.extraction_failed,
            'accuracy': f"{self.accuracy_percentage:.2f}%",
            'avg_error': f"{self.average_error:.2f}",
            'max_error': f"{self.max_error:.2f}",
        }


class OCRAccuracyEvaluator:
    def __init__(self, results_file: str, ground_truth_file: Optional[str] = None):
        self.results_file = Path(results_file)
        self.ground_truth_file = Path(ground_truth_file) if ground_truth_file else None
        self.results = []
        self.ground_truth = {}
        self.metrics = {}

        self._load_results()
        if self.ground_truth_file and self.ground_truth_file.exists():
            self._load_ground_truth()

    def _load_results(self):
        """Load OCR results from JSON file"""
        if not self.results_file.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_file}")

        with open(self.results_file, 'r', encoding='utf-8') as f:
            self.results = json.load(f)

    def _load_ground_truth(self):
        """Load manually verified ground truth amounts"""
        with open(self.ground_truth_file, 'r', encoding='utf-8') as f:
            self.ground_truth = json.load(f)

    def extract_amount_from_filename(self, filename: str) -> Optional[float]:
        """
        Try to extract amount from filename patterns
        e.g., "invoice_123.45.jpg" -> 123.45
        """
        # Look for number patterns in filename
        numbers = re.findall(r'(\d+\.?\d*)', filename)
        if numbers:
            # If file has amount in name, return the largest number
            return float(max(numbers, key=float))
        return None

    def evaluate_against_ground_truth(self) -> None:
        """Evaluate models against manually verified amounts"""
        if not self.ground_truth:
            print("No ground truth data available. Skipping ground truth evaluation.")
            return

        model_names = set()
        for result in self.results:
            model_names.update(result['results'].keys())

        metrics = {model: AccuracyMetrics(model) for model in model_names}

        for result in self.results:
            filename = result['filename']

            # Get ground truth amount
            truth_amount = self.ground_truth.get(filename) or result.get('manual_amount')
            if truth_amount is None:
                continue

            for model_name, model_data in result['results'].items():
                metrics[model_name].total_invoices += 1

                extracted = model_data.get('extracted_amount')
                if extracted is None:
                    metrics[model_name].extraction_failed += 1
                else:
                    error = extracted - truth_amount
                    if abs(error) < 0.01:  # Allow small rounding errors
                        metrics[model_name].correct_amounts += 1
                    else:
                        metrics[model_name].incorrect_amounts += 1
                        metrics[model_name].errors.append(error)

        # Calculate and display metrics
        for model_name in sorted(model_names):
            metrics[model_name].calculate()

        self._display_metrics(metrics)
        self._save_evaluation_report(metrics)

    def evaluate_model_consistency(self) -> None:
        """Evaluate consistency between models for the same invoices"""
        print("\n" + "="*80)
        print("MODEL CONSISTENCY ANALYSIS")
        print("="*80 + "\n")

        model_names = set()
        for result in self.results:
            model_names.update(result['results'].keys())

        if len(model_names) < 2:
            print("Need at least 2 models for consistency analysis.")
            return

        print(f"{'Invoice':<50} | {' | '.join(f'{m:<10}' for m in sorted(model_names))}")
        print("-"*80)

        agreement_count = 0
        total_invoices = 0

        for result in self.results:
            filename = result['filename'][:45]
            amounts = []

            for model in sorted(model_names):
                if model in result['results']:
                    amount = result['results'][model].get('extracted_amount')
                    amounts.append(amount)
                    print(f"{amount:<10}", end=' | ' if model != sorted(model_names)[-1] else '\n')
                else:
                    print(f"{'N/A':<10}", end=' | ')

            # Check if all models agree
            if amounts and len(set(amounts)) == 1:
                agreement_count += 1
            total_invoices += 1

        print("-"*80)
        if total_invoices > 0:
            agreement_percentage = (agreement_count / total_invoices) * 100
            print(f"\nModel agreement rate: {agreement_percentage:.2f}% ({agreement_count}/{total_invoices})")

    def _display_metrics(self, metrics: Dict[str, AccuracyMetrics]) -> None:
        """Display accuracy metrics in table format"""
        print("\n" + "="*80)
        print("ACCURACY EVALUATION RESULTS")
        print("="*80 + "\n")

        print(f"{'Model':<15} | {'Correct':<8} | {'Incorrect':<10} | {'Failed':<7} | {'Accuracy':<10}")
        print("-"*80)

        for model_name in sorted(metrics.keys()):
            m = metrics[model_name]
            print(f"{m.model_name:<15} | {m.correct_amounts:<8} | {m.incorrect_amounts:<10} | "
                  f"{m.extraction_failed:<7} | {m.accuracy_percentage:>7.2f}%")

        print("\n" + "="*80)
        print("ERROR STATISTICS")
        print("="*80 + "\n")

        print(f"{'Model':<15} | {'Avg Error':<12} | {'Max Error':<12}")
        print("-"*80)

        for model_name in sorted(metrics.keys()):
            m = metrics[model_name]
            print(f"{m.model_name:<15} | ¥{m.average_error:>10.2f} | ¥{m.max_error:>10.2f}")

    def _save_evaluation_report(self, metrics: Dict[str, AccuracyMetrics]) -> None:
        """Save evaluation report to JSON"""
        output_dir = Path("./ocr_results")
        output_dir.mkdir(exist_ok=True)

        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': [metrics[m].to_dict() for m in sorted(metrics.keys())]
        }

        report_file = output_dir / f"accuracy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\nEvaluation report saved to: {report_file}")


def main():
    # Try to find the latest results file
    results_dir = Path("./ocr_results")
    results_files = sorted(results_dir.glob("results_*.json"), reverse=True)

    if not results_files:
        print("No results files found. Run invoice_ocr_processor.py first.")
        return

    latest_results = results_files[0]
    print(f"Evaluating results from: {latest_results}\n")

    # Check for ground truth file
    ground_truth_file = Path("./ground_truth.json")

    evaluator = OCRAccuracyEvaluator(str(latest_results), str(ground_truth_file) if ground_truth_file.exists() else None)

    # Run evaluations
    if evaluator.ground_truth:
        evaluator.evaluate_against_ground_truth()
    else:
        print("Ground truth file not found. Running consistency analysis instead.\n")

    evaluator.evaluate_model_consistency()


if __name__ == "__main__":
    main()
