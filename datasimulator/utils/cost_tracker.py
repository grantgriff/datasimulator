"""
Cost tracking with automatic $20 limit and user prompts.

Monitors API usage costs and prompts user when limits are reached.
"""

import logging
from typing import Optional, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


class CostTracker:
    """
    Track API costs and enforce spending limits.

    Features:
    - Automatic stop at configured limit (default $20)
    - User prompt to continue with increased limit
    - Cost breakdown by operation type
    - Historical tracking
    """

    def __init__(
        self,
        max_cost: float = 20.0,
        increment: float = 20.0,
        interactive: bool = True
    ):
        """
        Initialize cost tracker.

        Args:
            max_cost: Maximum cost before prompting user (USD)
            increment: Amount to increase limit by when user continues (USD)
            interactive: Whether to prompt user (False for automated tests)
        """
        self.max_cost = max_cost
        self.increment = increment
        self.interactive = interactive

        self.total_cost = 0.0
        self.cost_history = []
        self.cost_by_operation = {
            "generation": 0.0,
            "verification": 0.0,
            "diversity": 0.0,
            "other": 0.0
        }

        self.start_time = datetime.now()
        self.stopped = False

    def add_cost(
        self,
        cost: float,
        operation: str = "generation",
        metadata: Optional[dict] = None
    ) -> bool:
        """
        Add cost and check if limit exceeded.

        Args:
            cost: Cost to add (USD)
            operation: Type of operation (generation, verification, diversity, other)
            metadata: Additional context about this cost

        Returns:
            True if generation should continue, False if stopped
        """
        if self.stopped:
            return False

        # Record cost
        self.total_cost += cost
        if operation in self.cost_by_operation:
            self.cost_by_operation[operation] += cost
        else:
            self.cost_by_operation["other"] += cost

        # Add to history
        self.cost_history.append({
            "timestamp": datetime.now(),
            "cost": cost,
            "operation": operation,
            "total_cost": self.total_cost,
            "metadata": metadata or {}
        })

        logger.debug(
            f"Added ${cost:.4f} for {operation}. "
            f"Total: ${self.total_cost:.2f}/${self.max_cost:.2f}"
        )

        # Check if limit exceeded
        if self.total_cost >= self.max_cost:
            return self._handle_limit_reached()

        return True

    def _handle_limit_reached(self) -> bool:
        """Handle when cost limit is reached."""
        logger.warning(f"Cost limit reached: ${self.total_cost:.2f}")

        if not self.interactive:
            logger.info("Non-interactive mode: stopping generation")
            self.stopped = True
            return False

        # Show cost breakdown
        print("\n" + "=" * 60)
        print(f"💰 COST LIMIT REACHED: ${self.total_cost:.2f} / ${self.max_cost:.2f}")
        print("=" * 60)
        print("\nCost Breakdown:")
        for operation, cost in self.cost_by_operation.items():
            if cost > 0:
                percentage = (cost / self.total_cost) * 100
                print(f"  {operation.capitalize():12s}: ${cost:7.2f} ({percentage:5.1f}%)")
        print(f"\n  {'Total':12s}: ${self.total_cost:7.2f}")

        elapsed = (datetime.now() - self.start_time).total_seconds()
        print(f"\nTime elapsed: {elapsed:.1f}s")
        print(f"Cost per minute: ${(self.total_cost / elapsed * 60):.2f}/min")
        print("=" * 60)

        # Ask user if they want to continue
        while True:
            response = input(
                f"\nContinue generation? This will increase limit by ${self.increment:.2f}. "
                "(y/n): "
            ).strip().lower()

            if response == 'y':
                self.max_cost += self.increment
                logger.info(f"Limit increased to ${self.max_cost:.2f}")
                print(f"✓ Limit increased to ${self.max_cost:.2f}\n")
                return True

            elif response == 'n':
                logger.info("User chose to stop generation")
                print("✓ Generation stopped by user\n")
                self.stopped = True
                return False

            else:
                print("Please enter 'y' or 'n'")

    def can_continue(self) -> bool:
        """Check if generation can continue."""
        return not self.stopped and self.total_cost < self.max_cost

    def get_remaining_budget(self) -> float:
        """Get remaining budget before limit."""
        return max(0, self.max_cost - self.total_cost)

    def get_summary(self) -> dict:
        """Get summary of costs."""
        elapsed = (datetime.now() - self.start_time).total_seconds()

        return {
            "total_cost": self.total_cost,
            "max_cost": self.max_cost,
            "remaining": self.get_remaining_budget(),
            "cost_by_operation": self.cost_by_operation.copy(),
            "time_elapsed_seconds": elapsed,
            "cost_per_minute": (self.total_cost / elapsed * 60) if elapsed > 0 else 0,
            "stopped": self.stopped,
            "num_operations": len(self.cost_history)
        }

    def print_summary(self):
        """Print detailed cost summary."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("📊 COST SUMMARY")
        print("=" * 60)
        print(f"Total Cost:        ${summary['total_cost']:.2f}")
        print(f"Budget Limit:      ${summary['max_cost']:.2f}")
        print(f"Remaining:         ${summary['remaining']:.2f}")
        print(f"\nTime Elapsed:      {summary['time_elapsed_seconds']:.1f}s")
        print(f"Cost per Minute:   ${summary['cost_per_minute']:.2f}/min")
        print(f"Total Operations:  {summary['num_operations']}")

        print("\nBreakdown by Operation:")
        for operation, cost in summary['cost_by_operation'].items():
            if cost > 0:
                percentage = (cost / summary['total_cost']) * 100
                print(f"  {operation.capitalize():12s}: ${cost:7.2f} ({percentage:5.1f}%)")

        print("=" * 60 + "\n")

    def reset(self):
        """Reset cost tracker."""
        self.total_cost = 0.0
        self.cost_history = []
        self.cost_by_operation = {
            "generation": 0.0,
            "verification": 0.0,
            "diversity": 0.0,
            "other": 0.0
        }
        self.start_time = datetime.now()
        self.stopped = False

    def export_history(self) -> list:
        """Export cost history for analysis."""
        return [
            {
                **entry,
                "timestamp": entry["timestamp"].isoformat()
            }
            for entry in self.cost_history
        ]
