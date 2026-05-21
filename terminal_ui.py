"""
Terminal UI module for RADMC-3D simulations
Includes basic helper functions and advanced progress tracking
"""

import os
import time
import psutil
import re
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, ProgressColumn, Task, TimeRemainingColumn
from rich.text import Text
from rich import box
from rich.panel import Panel

console = Console()

SUITE_NAME = "RADMC ProtoDisk Suite"
SUITE_VERSION = "v0.1-alpha"
SUITE_SUBTITLE = "RADMC-3D protoplanetary disk workflow"


# ===========================================================================
# HELPER: COLOR GRADIENT
# ===========================================================================

def get_gradient_color(percentage):
    """
    Calculates an RGB color on a gradient from Red -> Yellow -> Green
    based on a percentage (0.0 to 1.0).
    """
    p = max(0.0, min(1.0, percentage))
    
    # 0% to 50%: Red to Yellow
    if p < 0.5:
        segment_p = p * 2
        r = 255
        g = int(255 * segment_p)
        b = 0
    # 50% to 100%: Yellow to Green
    else:
        segment_p = (p - 0.5) * 2
        r = int(255 * (1 - segment_p))
        g = 255
        b = 0
        
    return f"rgb({r},{g},{b})"


# ===========================================================================
# BASIC UI FUNCTIONS
# ===========================================================================

def print_banner(mode, name, category, timestamp):
    """Print the suite banner and run identity."""
    console.print(
        Panel.fit(
            f"[bold cyan]{SUITE_NAME}[/bold cyan] [bold yellow]{SUITE_VERSION}[/bold yellow]\n"
            f"[white]{SUITE_SUBTITLE}[/white]",
            border_style="cyan",
            box=box.ASCII,
            padding=(1, 4),
        )
    )

    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column(style="cyan")
    info_table.add_column(style="white")
    
    info_table.add_row("Mode:", mode.upper())
    info_table.add_row("Name:", name)
    info_table.add_row("Category:", f"[bold]{category}[/bold]")
    info_table.add_row("Timestamp:", timestamp)
    info_table.add_row("Started:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    console.print(info_table)
    console.print()


def print_pre_run_summary(run_name, config_name, run_dir, ui_mode, seconds=5):
    """Show the resolved run setup briefly before RADMC starts."""
    output_folder = os.path.basename(run_dir.rstrip(os.sep)) or run_dir

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="cyan", no_wrap=True)
    table.add_column(style="white", no_wrap=True)

    table.add_row("Run", run_name)
    table.add_row("Config", config_name or "config.py")
    table.add_row("Folder", output_folder)
    table.add_row("UI", ui_mode)

    console.print(
        Panel.fit(
            table,
            title=f"[bold cyan]{SUITE_NAME}[/bold cyan] [bold yellow]{SUITE_VERSION}[/bold yellow]",
            subtitle=f"Starting in {seconds} seconds",
            border_style="bright_green",
            box=box.ASCII,
            padding=(1, 2),
        )
    )
    time.sleep(seconds)
    console.print()


def print_success(message):
    console.print(f"[bold bright_green]OK[/bold bright_green] [bright_green]{message}[/bright_green]")
    return


def print_warning(message):
    console.print(f"[yellow]WARN[/yellow] {message}")
    return


def print_error(message):
    console.print(f"[red]ERR[/red] {message}")
    return


def print_info(message):
    console.print(f"[cyan]->[/cyan] {message}")
    return


def print_separator():
    console.print("[dim]" + "-" * 60 + "[/dim]")
    return


def print_parameter_table(params, show_all=False):
    key_params = [
        'mdisk', 'hrdisk', 'plh', 'tstar', 'incl', 
        'h_spiral_amp', 'sig_spiral_amp', 'n_arms',
        'nphot', 'nphot_spec', 'threads'
    ]
    
    table = Table(title="Key Parameters", box=box.ROUNDED)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="white")
    
    if show_all:
        for key, value in params.items():
            table.add_row(key, str(value))
    else:
        for key in key_params:
            if key in params:
                table.add_row(key, str(params[key]))
    
    console.print(table)
    console.print()


def print_system_info():
    table = Table(title="System Information", box=box.ROUNDED, show_header=False)
    table.add_column(style="cyan")
    table.add_column(style="white")
    
    cpu_count = os.cpu_count()
    table.add_row("CPU Threads", str(cpu_count))
    
    mem = psutil.virtual_memory()
    mem_total_gb = mem.total / (1024**3)
    mem_avail_gb = mem.available / (1024**3)
    table.add_row("RAM Available", f"{mem_avail_gb:.1f} GB / {mem_total_gb:.1f} GB")
    
    disk = psutil.disk_usage('.')
    disk_free_gb = disk.free / (1024**3)
    table.add_row("Disk Free", f"{disk_free_gb:.1f} GB")
    
    console.print(table)
    console.print()


# ===========================================================================
# CUSTOM PROGRESS COLUMNS
# ===========================================================================

class ZeroPulseBarColumn(BarColumn):
    """
    A special bar column that behaves like a standard progress bar,
    BUT if 'completed' is 0 (or total is None), it shows the 
    'Pulse/Scanner' animation instead of an empty bar.
    """
    def render(self, task: Task):
        # Unknown totals or zero progress use the pulse animation.
        # This keeps the bar visibly active before the first update.
        if task.total is None or task.completed == 0:
            return super().render(task_copy_with_total_none(task))
        
        # Once progress exists, render the normal bar.
        return super().render(task)

def task_copy_with_total_none(task):
    """Helper to create a temporary task view with total=None."""
    import copy
    new_task = copy.copy(task)
    new_task.total = None 
    return new_task


class SmartPercentageColumn(ProgressColumn):
    """
    Renders percentage only if total is known.
    """
    def render(self, task):
        if task.total is None:
            return Text(" -- ", style="dim")
        return Text(f"{task.percentage:>3.0f}%", style="blue")


class DynamicCountColumn(ProgressColumn):
    """
    Renders the completed/total count with gradient colors.
    If total is known, handles gradient. If not, shows 'Computing...'
    """
    def render(self, task):
        # Fall 1: Total komplett unbekannt (echtes Raytracing ohne Photonen)
        if task.total is None:
            dots = (int(time.time() * 2) % 3) + 1
            return Text(f"Computing{'.' * dots}", style="bold cyan")
            
        # Fall 2: Total ist bekannt (auch wenn completed noch 0 ist)
        completed = int(task.completed)
        total = int(task.total)
        text_str = f"{completed}/{total}"
        
        if task.total > 0:
            percentage = task.completed / task.total
        else:
            percentage = 0
        
        color = get_gradient_color(percentage)
        
        if percentage >= 1.0:
            return Text(text_str, style="bold white on green")
            
        return Text(text_str, style=f"bold {color}")


# ===========================================================================
# ADVANCED PROGRESS TRACKER
# ===========================================================================

class AdvancedPhaseTracker:
    """
    Advanced phase tracker with progress bars and real-time updates.
    """
    
    def __init__(self, phases, estimated_times=None, max_log_lines=12, update_interval=1000):
        self.phases = phases
        self.estimated_times = estimated_times or {}
        self.current_phase_idx = -1
        self.start_time = time.time()
        self.phase_start_time = None
        self.phase_times = {}
        
        self.update_interval = update_interval
        self.last_reported_step = 0
        
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            
            # Pulse style for the active phase bar.
            ZeroPulseBarColumn(bar_width=40, pulse_style="bold bright_green"),
            
            SmartPercentageColumn(),
            DynamicCountColumn(), 
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
            refresh_per_second=10 
        )
        
        self.overall_task = self.progress.add_task("Total", total=len(phases))
        self.phase_task = self.progress.add_task("Waiting...", total=None, visible=False)

    def start(self):
        self.progress.start()

    def stop(self):
        self.progress.stop()

    def log(self, message):
        if "[" in message and "]" not in message:
            message = message.replace("[", "\\[")

        task = self.progress.tasks[self.phase_task]
        total = task.total

        if total and total > 0:
            def color_number_match(match):
                num_str = match.group(0)
                try:
                    val = int(num_str)
                    pct = val / total
                    if pct > 1.05: return num_str
                    color = get_gradient_color(pct)
                    return f"[bold {color}]{num_str}[/]"
                except ValueError:
                    return num_str

            message = re.sub(r'\b\d+\b', color_number_match, message)

        self.progress.console.print(f"  {message}")

    def set_phase_total(self, total_steps):
        """Sets maximum for current phase. Updates total immediately."""
        self.progress.update(self.phase_task, total=total_steps, completed=0)
        self.last_reported_step = 0

    def update_progress(self, step, force=False):
        """Update progress."""
        task = self.progress.tasks[self.phase_task]
        
        if task.total is not None:
            small_total = task.total <= self.update_interval
            if force or small_total or step - self.last_reported_step >= self.update_interval:
                self.progress.update(self.phase_task, completed=step)
                self.last_reported_step = step
                
            if step >= task.total:
                 self.progress.update(self.phase_task, completed=step)

    def start_phase(self, phase_name):
        self.current_phase_idx = self.phases.index(phase_name)
        self.phase_start_time = time.time()
        
        self.progress.update(self.overall_task, completed=self.current_phase_idx)
        
        estimated = self.estimated_times.get(phase_name)
        desc = f"[yellow]{phase_name}[/yellow]"
        if estimated:
            desc += f" (~{estimated} min)"
            
        self.progress.reset(self.phase_task)
        self.progress.update(self.phase_task, description=desc, total=None, visible=True)
        
        self.log(f"[bold bright_green]->[/bold bright_green] Starting: [bold bright_green]{phase_name}[/bold bright_green]")

    def complete_phase(self, phase_name):
        if self.phase_start_time:
            duration = time.time() - self.phase_start_time
            self.phase_times[phase_name] = duration
            duration_str = f"{int(duration)}s"
            self.log(f"[bold bright_green]OK[/bold bright_green] Done: [bold]{phase_name}[/bold] [bright_green]({duration_str})[/bright_green]")
        
        task = self.progress.tasks[self.phase_task]
        final_total = task.total if task.total else 100
        
        self.progress.update(self.phase_task, total=final_total, completed=final_total)
        self.progress.update(self.overall_task, completed=self.current_phase_idx + 1)

    def get_total_time(self):
        return time.time() - self.start_time

    def print_summary(self):
        console.print("\n[bold]Summary:[/bold]")
        table = Table(box=None, show_header=False)
        table.add_column(style="cyan")
        table.add_column(style="white")
        
        for phase in self.phases:
            d = self.phase_times.get(phase, 0)
            if phase in self.phase_times:
                table.add_row(phase, f"{d:.1f}s")
            else:
                table.add_row(phase, "skipped")
                
        console.print(table)
