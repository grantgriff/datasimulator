"""
Human review interface for manual sample inspection.

Provides interactive CLI for reviewing and approving/rejecting samples.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax

logger = logging.getLogger(__name__)


class HumanReviewer:
    """
    Interactive human review interface.

    Allows manual inspection and approval/rejection of generated samples.
    """

    def __init__(self, interactive: bool = True):
        """
        Initialize human reviewer.

        Args:
            interactive: Whether to enable interactive mode
        """
        self.interactive = interactive
        self.console = Console()

        # Review statistics
        self.stats = {
            "total_reviewed": 0,
            "approved": 0,
            "rejected": 0,
            "skipped": 0
        }

    def review_batch(
        self,
        samples: List[Dict[str, Any]],
        show_quality_scores: bool = True
    ) -> Tuple[List[Dict[str, Any]], List[int]]:
        """
        Review a batch of samples interactively.

        Args:
            samples: List of samples to review
            show_quality_scores: Whether to show quality scores

        Returns:
            Tuple of (approved_samples, rejected_indices)
        """
        if not self.interactive:
            logger.info("Non-interactive mode: auto-approving all samples")
            return samples, []

        self.console.print("\n" + "="*60)
        self.console.print("[bold cyan]🔍 Human Review Session[/bold cyan]")
        self.console.print("="*60)
        self.console.print(f"Reviewing {len(samples)} samples\n")
        self.console.print("[dim]Commands: (a)pprove, (r)eject, (s)kip, (q)uit[/dim]\n")

        approved = []
        rejected_indices = []

        for i, sample in enumerate(samples):
            self.stats["total_reviewed"] += 1

            # Display sample
            self._display_sample(i + 1, len(samples), sample, show_quality_scores)

            # Get user decision
            decision = self._get_user_decision()

            if decision == "approve":
                approved.append(sample)
                self.stats["approved"] += 1
                self.console.print("[green]✓ Approved[/green]\n")

            elif decision == "reject":
                rejected_indices.append(i)
                self.stats["rejected"] += 1
                reason = Prompt.ask("[yellow]Rejection reason (optional)[/yellow]", default="")
                if reason:
                    sample["rejection_reason"] = reason
                self.console.print("[red]✗ Rejected[/red]\n")

            elif decision == "skip":
                self.stats["skipped"] += 1
                self.console.print("[dim]⊘ Skipped[/dim]\n")
                continue

            elif decision == "quit":
                self.console.print("[yellow]Review session ended by user[/yellow]\n")
                break

        # Show summary
        self._show_summary()

        return approved, rejected_indices

    def _display_sample(
        self,
        current: int,
        total: int,
        sample: Dict[str, Any],
        show_quality_scores: bool
    ):
        """Display a sample for review."""
        # Header
        self.console.print(f"\n[bold]Sample {current}/{total}[/bold]")
        self.console.print("-" * 60)

        # Quality scores (if available)
        if show_quality_scores and "metrics" in sample:
            metrics = sample["metrics"]
            quality_score = metrics.get("quality_score", "N/A")

            table = Table(show_header=False, box=None)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Quality Score", f"{quality_score:.1f}/10")

            if "dimension_scores" in metrics:
                dim_scores = metrics["dimension_scores"]
                for dim, score in dim_scores.items():
                    if isinstance(score, (int, float)):
                        table.add_row(f"  {dim.title()}", f"{score:.1f}")

            self.console.print(table)
            self.console.print()

        # Sample content
        data = sample.get("data", sample)

        # Format based on data structure
        if "messages" in data:
            # SFT messages format
            self._display_messages(data["messages"])

        elif "prompt" in data and "completion" in data:
            # SFT completion format
            self._display_completion(data["prompt"], data["completion"])

        elif "prompt" in data and "chosen" in data:
            # DPO format
            self._display_dpo(data)

        else:
            # Generic display
            self._display_generic(data)

    def _display_messages(self, messages: List[Dict[str, str]]):
        """Display messages format."""
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if role == "system":
                color = "blue"
                icon = "⚙️"
            elif role == "user":
                color = "cyan"
                icon = "👤"
            elif role == "assistant":
                color = "green"
                icon = "🤖"
            else:
                color = "white"
                icon = "•"

            self.console.print(
                Panel(
                    content,
                    title=f"{icon} {role.title()}",
                    border_style=color,
                    padding=(0, 1)
                )
            )

    def _display_completion(self, prompt: str, completion: str):
        """Display completion format."""
        self.console.print(
            Panel(
                prompt,
                title="📝 Prompt",
                border_style="cyan",
                padding=(0, 1)
            )
        )
        self.console.print(
            Panel(
                completion,
                title="💬 Completion",
                border_style="green",
                padding=(0, 1)
            )
        )

    def _display_dpo(self, data: Dict[str, Any]):
        """Display DPO format."""
        self.console.print(
            Panel(
                data.get("prompt", ""),
                title="📝 Prompt",
                border_style="cyan",
                padding=(0, 1)
            )
        )
        self.console.print(
            Panel(
                data.get("chosen", ""),
                title="✅ Chosen (Better)",
                border_style="green",
                padding=(0, 1)
            )
        )
        self.console.print(
            Panel(
                data.get("rejected", ""),
                title="❌ Rejected (Worse)",
                border_style="red",
                padding=(0, 1)
            )
        )

    def _display_generic(self, data: Dict[str, Any]):
        """Display generic data format."""
        syntax = Syntax(
            json.dumps(data, indent=2),
            "json",
            theme="monokai",
            line_numbers=False
        )
        self.console.print(
            Panel(
                syntax,
                title="📄 Sample Data",
                border_style="blue",
                padding=(0, 1)
            )
        )

    def _get_user_decision(self) -> str:
        """Get user's decision on sample."""
        while True:
            choice = Prompt.ask(
                "[bold]Decision[/bold]",
                choices=["a", "r", "s", "q"],
                default="a"
            )

            if choice == "a":
                return "approve"
            elif choice == "r":
                return "reject"
            elif choice == "s":
                return "skip"
            elif choice == "q":
                if Confirm.ask("[yellow]Are you sure you want to quit?[/yellow]"):
                    return "quit"
                else:
                    continue

    def _show_summary(self):
        """Show review session summary."""
        self.console.print("\n" + "="*60)
        self.console.print("[bold cyan]📊 Review Summary[/bold cyan]")
        self.console.print("="*60)

        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Count", style="green", width=10)
        table.add_column("Percentage", style="yellow")

        total = self.stats["total_reviewed"]

        table.add_row(
            "Total Reviewed",
            str(total),
            "100%"
        )
        table.add_row(
            "Approved",
            str(self.stats["approved"]),
            f"{self.stats['approved']/total*100:.1f}%" if total > 0 else "0%"
        )
        table.add_row(
            "Rejected",
            str(self.stats["rejected"]),
            f"{self.stats['rejected']/total*100:.1f}%" if total > 0 else "0%"
        )
        table.add_row(
            "Skipped",
            str(self.stats["skipped"]),
            f"{self.stats['skipped']/total*100:.1f}%" if total > 0 else "0%"
        )

        self.console.print(table)
        self.console.print("="*60 + "\n")

    def quick_review(
        self,
        sample: Dict[str, Any],
        auto_approve_above: float = 8.0
    ) -> bool:
        """
        Quick review with auto-approval for high-quality samples.

        Args:
            sample: Sample to review
            auto_approve_above: Auto-approve if quality score above this

        Returns:
            True if approved, False if rejected
        """
        # Check for auto-approval
        quality_score = sample.get("metrics", {}).get("quality_score", 0.0)

        if quality_score >= auto_approve_above:
            logger.debug(f"Auto-approved: quality={quality_score:.1f}")
            return True

        # Manual review
        if not self.interactive:
            # Non-interactive: approve if above threshold
            return quality_score >= 6.0

        # Interactive review
        self._display_sample(1, 1, sample, show_quality_scores=True)
        decision = self._get_user_decision()

        return decision == "approve"

    def reset_stats(self):
        """Reset review statistics."""
        self.stats = {
            "total_reviewed": 0,
            "approved": 0,
            "rejected": 0,
            "skipped": 0
        }

    def get_stats(self) -> Dict[str, int]:
        """Get review statistics."""
        return self.stats.copy()
