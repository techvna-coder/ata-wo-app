# core/progress_tracker.py
"""
Progress tracking for WO processing with multiple display options:
- Console progress bar (for scripts)
- Streamlit progress bar (for UI)
- Callback-based (for custom integration)
- File-based (for background jobs)
"""
from __future__ import annotations
import time
from typing import Optional, Callable, Dict, Any
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ProgressState:
    """Current progress state."""
    total: int = 0
    processed: int = 0
    success: int = 0
    failed: int = 0
    skipped: int = 0
    current_item: str = ""
    start_time: float = 0.0
    status: str = "idle"  # idle, running, completed, error
    
    @property
    def percentage(self) -> float:
        """Progress percentage (0-100)."""
        if self.total == 0:
            return 0.0
        return (self.processed / self.total) * 100
    
    @property
    def elapsed_seconds(self) -> float:
        """Elapsed time in seconds."""
        if self.start_time == 0:
            return 0.0
        return time.time() - self.start_time
    
    @property
    def eta_seconds(self) -> float:
        """Estimated time remaining in seconds."""
        if self.processed == 0 or self.total == 0:
            return 0.0
        rate = self.processed / self.elapsed_seconds
        remaining = self.total - self.processed
        return remaining / rate if rate > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            'percentage': self.percentage,
            'elapsed_seconds': self.elapsed_seconds,
            'eta_seconds': self.eta_seconds,
        }


# ============================================================================
# BASE TRACKER
# ============================================================================
class ProgressTracker:
    """Base progress tracker."""
    
    def __init__(self, total: int = 0, description: str = "Processing"):
        self.state = ProgressState(total=total, status="idle")
        self.description = description
        self.callbacks: list[Callable] = []
    
    def start(self):
        """Start tracking."""
        self.state.start_time = time.time()
        self.state.status = "running"
        self._notify()
    
    def update(
        self,
        increment: int = 1,
        success: bool = True,
        current_item: str = "",
        skipped: bool = False
    ):
        """Update progress."""
        self.state.processed += increment
        
        if skipped:
            self.state.skipped += increment
        elif success:
            self.state.success += increment
        else:
            self.state.failed += increment
        
        if current_item:
            self.state.current_item = current_item
        
        self._notify()
    
    def complete(self):
        """Mark as completed."""
        self.state.status = "completed"
        self._notify()
    
    def error(self, message: str = ""):
        """Mark as error."""
        self.state.status = "error"
        self.state.current_item = message
        self._notify()
    
    def add_callback(self, callback: Callable[[ProgressState], None]):
        """Add callback to be called on updates."""
        self.callbacks.append(callback)
    
    def _notify(self):
        """Notify all callbacks."""
        for callback in self.callbacks:
            try:
                callback(self.state)
            except Exception:
                pass  # Don't let callback errors break tracking


# ============================================================================
# CONSOLE PROGRESS BAR
# ============================================================================
class ConsoleProgressTracker(ProgressTracker):
    """Progress tracker with console output."""
    
    def __init__(self, total: int, description: str = "Processing", width: int = 50):
        super().__init__(total, description)
        self.width = width
        self.last_print_time = 0
        self.update_interval = 0.1  # Update every 100ms
    
    def _notify(self):
        """Print progress bar to console."""
        super()._notify()
        
        # Throttle updates
        now = time.time()
        if now - self.last_print_time < self.update_interval and self.state.status == "running":
            return
        self.last_print_time = now
        
        # Build progress bar
        pct = self.state.percentage
        filled = int(self.width * pct / 100)
        bar = "█" * filled + "░" * (self.width - filled)
        
        # Time info
        elapsed = self.state.elapsed_seconds
        eta = self.state.eta_seconds
        
        # Status line
        status_parts = []
        if self.state.success > 0:
            status_parts.append(f"✓{self.state.success}")
        if self.state.failed > 0:
            status_parts.append(f"✗{self.state.failed}")
        if self.state.skipped > 0:
            status_parts.append(f"⊘{self.state.skipped}")
        
        status_str = " ".join(status_parts) if status_parts else ""
        
        # Current item (truncate if too long)
        current = self.state.current_item[:40] + "..." if len(self.state.current_item) > 40 else self.state.current_item
        
        # Print (overwrite previous line)
        line = f"\r{self.description}: |{bar}| {pct:5.1f}% ({self.state.processed}/{self.state.total}) "
        line += f"[{elapsed:.0f}s<{eta:.0f}s] {status_str}"
        if current:
            line += f" | {current}"
        
        print(line, end="", flush=True)
        
        # New line on completion
        if self.state.status in ["completed", "error"]:
            print()


# ============================================================================
# STREAMLIT PROGRESS
# ============================================================================
class StreamlitProgressTracker(ProgressTracker):
    """Progress tracker for Streamlit UI."""
    
    def __init__(self, total: int, description: str = "Processing"):
        super().__init__(total, description)
        self.progress_bar = None
        self.status_text = None
        self.metrics_cols = None
        
    def initialize_ui(self, st_container=None):
        """Initialize Streamlit UI components."""
        try:
            import streamlit as st
            
            container = st_container or st
            
            # Progress bar
            self.progress_bar = container.progress(0)
            
            # Status text
            self.status_text = container.empty()
            
            # Metrics
            self.metrics_cols = container.columns(4)
            
        except ImportError:
            raise ImportError("streamlit not installed. pip install streamlit")
    
    def _notify(self):
        """Update Streamlit UI."""
        super()._notify()
        
        if not self.progress_bar:
            return
        
        try:
            import streamlit as st
            
            # Update progress bar
            progress_value = self.state.percentage / 100
            self.progress_bar.progress(progress_value)
            
            # Update status text
            elapsed = self.state.elapsed_seconds
            eta = self.state.eta_seconds
            
            status_msg = f"**{self.description}:** {self.state.processed}/{self.state.total} "
            status_msg += f"({self.state.percentage:.1f}%) - "
            status_msg += f"Elapsed: {elapsed:.0f}s, ETA: {eta:.0f}s"
            
            if self.state.current_item:
                status_msg += f"\n\n*Processing:* `{self.state.current_item}`"
            
            self.status_text.markdown(status_msg)
            
            # Update metrics
            if self.metrics_cols:
                self.metrics_cols[0].metric("✓ Success", self.state.success)
                self.metrics_cols[1].metric("✗ Failed", self.state.failed)
                self.metrics_cols[2].metric("⊘ Skipped", self.state.skipped)
                self.metrics_cols[3].metric("⏱ Speed", f"{self.state.processed/elapsed:.1f}/s" if elapsed > 0 else "0/s")
            
        except Exception as e:
            # Silently fail if UI update fails
            pass


# ============================================================================
# FILE-BASED TRACKER (for background jobs)
# ============================================================================
class FileProgressTracker(ProgressTracker):
    """Progress tracker that writes to file for monitoring."""
    
    def __init__(self, total: int, description: str = "Processing", output_file: str = "progress.json"):
        super().__init__(total, description)
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    def _notify(self):
        """Write progress to file."""
        super()._notify()
        
        try:
            data = {
                'description': self.description,
                'timestamp': datetime.now().isoformat(),
                **self.state.to_dict()
            }
            
            with open(self.output_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception:
            pass  # Don't let file write errors break tracking


def read_progress_file(filepath: str = "progress.json") -> Optional[Dict[str, Any]]:
    """Read progress from file."""
    try:
        with open(filepath) as f:
            return json.load(f)
    except Exception:
        return None


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================
def create_progress_tracker(
    total: int,
    description: str = "Processing",
    mode: str = "console"  # console, streamlit, file, silent
) -> ProgressTracker:
    """
    Create appropriate progress tracker based on mode.
    
    Args:
        total: Total items to process
        description: Description text
        mode: 'console', 'streamlit', 'file', or 'silent'
    
    Returns:
        ProgressTracker instance
    """
    if mode == "console":
        return ConsoleProgressTracker(total, description)
    elif mode == "streamlit":
        tracker = StreamlitProgressTracker(total, description)
        tracker.initialize_ui()
        return tracker
    elif mode == "file":
        return FileProgressTracker(total, description)
    else:  # silent
        return ProgressTracker(total, description)


# ============================================================================
# CONTEXT MANAGER
# ============================================================================
class track_progress:
    """Context manager for progress tracking."""
    
    def __init__(
        self,
        items,
        description: str = "Processing",
        mode: str = "console"
    ):
        self.items = list(items)
        self.description = description
        self.mode = mode
        self.tracker = None
    
    def __enter__(self):
        self.tracker = create_progress_tracker(
            len(self.items),
            self.description,
            self.mode
        )
        self.tracker.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.tracker.complete()
        else:
            self.tracker.error(str(exc_val))
        return False
    
    def __iter__(self):
        """Iterate with progress tracking."""
        for item in self.items:
            yield item
            self.tracker.update(current_item=str(item))


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    import random
    
    print("Demo 1: Console Progress Bar")
    print("-" * 70)
    
    # Console tracker
    tracker = ConsoleProgressTracker(100, "Processing WO records")
    tracker.start()
    
    for i in range(100):
        time.sleep(0.05)  # Simulate work
        
        # Random success/failure
        success = random.random() > 0.1
        skipped = random.random() > 0.9
        
        tracker.update(
            success=success,
            skipped=skipped,
            current_item=f"WO-2024-{i:04d}"
        )
    
    tracker.complete()
    
    print("\n\nDemo 2: Context Manager")
    print("-" * 70)
    
    # Using context manager
    items = [f"WO-2024-{i:04d}" for i in range(50)]
    
    with track_progress(items, "Batch processing", mode="console") as progress:
        for item in progress:
            time.sleep(0.02)
            # Processing logic here
    
    print("\n\nDemo 3: File-based Progress")
    print("-" * 70)
    
    # File tracker
    file_tracker = FileProgressTracker(30, "Background job", "demo_progress.json")
    file_tracker.start()
    
    for i in range(30):
        time.sleep(0.1)
        file_tracker.update(current_item=f"Item {i+1}")
        
        # Check progress file
        if i % 10 == 0:
            progress = read_progress_file("demo_progress.json")
            print(f"  Progress file: {progress['percentage']:.1f}% complete")
    
    file_tracker.complete()
    
    print("\n✓ All demos completed!")
