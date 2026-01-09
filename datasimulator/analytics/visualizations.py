"""
Analytics visualizations for dataset quality and diversity.

Provides rich console visualizations and statistics.
"""

import logging
from typing import List, Dict, Any, Optional
from collections import Counter
import statistics

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.layout import Layout
from rich.tree import Tree

logger = logging.getLogger(__name__)


class DatasetAnalytics:
    """
    Analytics and visualizations for generated datasets.

    Provides detailed statistics and visual reports.
    """

    def __init__(self):
        """Initialize analytics."""
        self.console = Console()

    def show_quality_report(
        self,
        samples: List[Dict[str, Any]],
        title: str = "Quality Report"
    ):
        """
        Show detailed quality report.

        Args:
            samples: List of samples with quality metrics
            title: Report title
        """
        if not samples:
            self.console.print("[yellow]No samples to analyze[/yellow]")
            return

        # Extract quality scores
        quality_scores = [
            s.get("metrics", {}).get("quality_score", 0.0)
            for s in samples
        ]

        # Calculate statistics
        stats = {
            "count": len(quality_scores),
            "mean": statistics.mean(quality_scores),
            "median": statistics.median(quality_scores),
            "stdev": statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0,
            "min": min(quality_scores),
            "max": max(quality_scores)
        }

        # Create report
        self.console.print(f"\n[bold cyan]{title}[/bold cyan]")
        self.console.print("=" * 60)

        # Summary table
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="green", width=15)

        table.add_row("Total Samples", str(stats["count"]))
        table.add_row("Mean Quality", f"{stats['mean']:.2f}/10")
        table.add_row("Median Quality", f"{stats['median']:.2f}/10")
        table.add_row("Std Deviation", f"{stats['stdev']:.2f}")
        table.add_row("Min Quality", f"{stats['min']:.2f}/10")
        table.add_row("Max Quality", f"{stats['max']:.2f}/10")

        self.console.print(table)

        # Quality distribution
        self._show_quality_distribution(quality_scores)

        # Dimension breakdown (if available)
        self._show_dimension_breakdown(samples)

        self.console.print("=" * 60 + "\n")

    def _show_quality_distribution(self, scores: List[float]):
        """Show distribution of quality scores."""
        # Create bins
        bins = {
            "Excellent (9-10)": 0,
            "Good (7-9)": 0,
            "Acceptable (6-7)": 0,
            "Below Threshold (<6)": 0
        }

        for score in scores:
            if score >= 9:
                bins["Excellent (9-10)"] += 1
            elif score >= 7:
                bins["Good (7-9)"] += 1
            elif score >= 6:
                bins["Acceptable (6-7)"] += 1
            else:
                bins["Below Threshold (<6)"] += 1

        # Display distribution
        self.console.print("\n[bold]Quality Distribution:[/bold]")

        table = Table(show_header=False, box=None)
        table.add_column("Range", style="cyan", width=25)
        table.add_column("Count", style="green", width=10)
        table.add_column("Bar", width=30)

        max_count = max(bins.values()) if bins.values() else 1

        for range_name, count in bins.items():
            percentage = count / len(scores) * 100
            bar_length = int((count / max_count) * 20)
            bar = "█" * bar_length

            color = "green"
            if "Below" in range_name:
                color = "red"
            elif "Acceptable" in range_name:
                color = "yellow"

            table.add_row(
                range_name,
                f"{count} ({percentage:.1f}%)",
                f"[{color}]{bar}[/{color}]"
            )

        self.console.print(table)

    def _show_dimension_breakdown(self, samples: List[Dict[str, Any]]):
        """Show breakdown by quality dimensions."""
        # Collect dimension scores
        dimensions = {}

        for sample in samples:
            dim_scores = sample.get("metrics", {}).get("dimension_scores", {})

            for dim, score in dim_scores.items():
                if isinstance(score, (int, float)):
                    if dim not in dimensions:
                        dimensions[dim] = []
                    dimensions[dim].append(score)

        if not dimensions:
            return

        self.console.print("\n[bold]Dimension Scores:[/bold]")

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Dimension", style="cyan", width=20)
        table.add_column("Mean", style="green", width=10)
        table.add_column("Min", style="yellow", width=10)
        table.add_column("Max", style="green", width=10)

        for dim, scores in sorted(dimensions.items()):
            table.add_row(
                dim.title(),
                f"{statistics.mean(scores):.2f}",
                f"{min(scores):.2f}",
                f"{max(scores):.2f}"
            )

        self.console.print(table)

    def show_diversity_report(
        self,
        diversity_score: float,
        cluster_info: Optional[Dict[int, List[Any]]] = None
    ):
        """
        Show diversity analysis report.

        Args:
            diversity_score: Overall diversity score (0-1)
            cluster_info: Optional cluster information
        """
        self.console.print("\n[bold cyan]Diversity Report[/bold cyan]")
        self.console.print("=" * 60)

        # Overall diversity
        color = "green" if diversity_score > 0.7 else "yellow" if diversity_score > 0.5 else "red"

        self.console.print(f"\n[bold]Overall Diversity:[/bold] [{color}]{diversity_score:.3f}[/{color}]")

        # Interpretation
        if diversity_score > 0.8:
            interpretation = "Excellent diversity - samples are highly varied"
        elif diversity_score > 0.6:
            interpretation = "Good diversity - samples have reasonable variation"
        elif diversity_score > 0.4:
            interpretation = "Moderate diversity - some repetition present"
        else:
            interpretation = "Low diversity - samples may be too similar"

        self.console.print(f"[dim]{interpretation}[/dim]\n")

        # Cluster information
        if cluster_info:
            self.console.print("[bold]Sample Clusters:[/bold]")

            tree = Tree("📊 Dataset Clusters")

            for cluster_id, samples in sorted(cluster_info.items()):
                cluster_node = tree.add(
                    f"Cluster {cluster_id}: {len(samples)} samples "
                    f"({len(samples)/sum(len(s) for s in cluster_info.values())*100:.1f}%)"
                )

            self.console.print(tree)

        self.console.print("=" * 60 + "\n")

    def show_generation_summary(
        self,
        total_samples: int,
        target_samples: int,
        total_cost: float,
        generation_time: float,
        quality_stats: Dict[str, float],
        refinement_stats: Optional[Dict[str, int]] = None
    ):
        """
        Show comprehensive generation summary.

        Args:
            total_samples: Total samples generated
            target_samples: Target sample count
            total_cost: Total cost in USD
            generation_time: Total time in seconds
            quality_stats: Quality statistics
            refinement_stats: Optional refinement statistics
        """
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )

        # Header
        header_text = f"[bold cyan]Generation Complete![/bold cyan] {total_samples}/{target_samples} samples"
        layout["header"].update(Panel(header_text, style="bold green"))

        # Body
        body_table = Table(show_header=False, box=None, padding=(0, 2))
        body_table.add_column("Metric", style="cyan", width=25)
        body_table.add_column("Value", style="green")

        # Statistics
        body_table.add_row("Total Samples", str(total_samples))
        body_table.add_row("Success Rate", f"{total_samples/target_samples*100:.1f}%")
        body_table.add_row("Average Quality", f"{quality_stats.get('mean', 0):.2f}/10")
        body_table.add_row("Quality Range", f"{quality_stats.get('min', 0):.1f} - {quality_stats.get('max', 0):.1f}")
        body_table.add_row("Total Cost", f"${total_cost:.2f}")
        body_table.add_row("Cost per Sample", f"${total_cost/total_samples:.3f}")
        body_table.add_row("Generation Time", f"{generation_time:.1f}s")
        body_table.add_row("Samples per Minute", f"{total_samples/(generation_time/60):.1f}")

        if refinement_stats:
            body_table.add_row("", "")
            body_table.add_row("[bold]Refinement Stats:[/bold]", "")
            body_table.add_row("  Samples Refined", str(refinement_stats.get("samples_refined", 0)))
            body_table.add_row("  Total Retries", str(refinement_stats.get("total_retries", 0)))
            body_table.add_row("  Success Rate", f"{refinement_stats.get('success_rate', 0)*100:.1f}%")

        layout["body"].update(body_table)

        # Footer
        footer_text = "💾 Ready to save!"
        layout["footer"].update(Panel(footer_text, style="dim"))

        self.console.print(layout)

    def create_progress_bar(
        self,
        total: int,
        description: str = "Generating"
    ) -> tuple[Progress, Any]:
        """
        Create a progress bar for generation.

        Args:
            total: Total items
            description: Progress description

        Returns:
            Tuple of (Progress object, task_id)
        """
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[cyan]{task.completed}/{task.total}[/cyan]"),
            console=self.console
        )

        task_id = progress.add_task(description, total=total)

        return progress, task_id
