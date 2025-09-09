"""TunaTale command-line interface main module."""
import asyncio
import logging
from pathlib import Path
from typing import Optional, Any, Dict

import typer
from rich.console import Console

from typing_extensions import Annotated

from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from typing_extensions import Annotated

from tunatale.core.parsers.lesson_parser import parse_lesson_file
from tunatale.infrastructure.factories import create_lesson_processor

# Basic logging and console setup
logging.basicConfig(level=logging.WARNING, format="%(message)s", datefmt="[%X]")
logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(
    name="tunatale",
    help="TunaTale - A tool to generate audio lessons for language learning.",
    add_completion=False,
    no_args_is_help=True,
)

def print_error(message: str):
    """Prints an error message to the console."""
    console.print(f"[bold red]Error:[/bold red] {message}")

def print_success(message: str):
    """Prints a success message to the console."""
    console.print(f"[bold green]✓[/bold green] {message}")

def print_info(message: str):
    """Prints an informational message to the console."""
    console.print(f"[bold blue]i[/bold blue] {message}")


class RichProgressAdapter:
    """An adapter to connect the LessonProcessor's progress calls to a Rich progress bar."""

    def __init__(self, progress: Progress):
        self.progress = progress
        self.overall_task: Optional[Any] = None
        self.step_task: Optional[Any] = None

    async def update(
        self,
        task_id: str,
        completed: int,
        total: int,
        status: str = "Processing...",
        **kwargs,
    ):
        """Update a progress task or create it if it doesn't exist."""
        description = f"[cyan]{status}[/cyan]"

        if task_id.startswith("lesson_"):
            if self.overall_task is None:
                self.overall_task = self.progress.add_task(
                    "[bold green]Overall Progress", total=total
                )
            self.progress.update(self.overall_task, completed=completed)
        else:
            if self.step_task is None:
                self.step_task = self.progress.add_task(description, total=total)
            else:
                self.progress.update(
                    self.step_task,
                    total=total,
                    completed=completed,
                    description=description,
                    visible=True,
                )

    async def complete_task(self, task_id: str):
        """Mark a task as complete."""
        if task_id.startswith("lesson_") and self.overall_task is not None:
            self.progress.update(
                self.overall_task, completed=self.progress.tasks[self.overall_task].total
            )
        elif self.step_task is not None:
            # When a step completes, hide it by making it invisible
            self.progress.update(
                self.step_task,
                completed=self.progress.tasks[self.step_task].total,
                visible=False,
            )


async def run_lesson_processing(input_file: Path, output_dir: Path):
    """The core async function to process a lesson."""
    logger.info(f"Processing lesson: {input_file.name}")
    logger.info(f"Output will be saved to: {output_dir}")

    # Hardcoded default configuration, replacing the need for config files.
    config = {
        "tts": {
            "provider": "multi",
            "edge_tts": {},
            "gtts": {},
        },
        "audio": {
            "output_format": "mp3",
            "silence_between_phrases": 0.5,
            "silence_between_sections": 1.0,
            "normalize_audio": True,
            "trim_silence": True,
            "cleanup_temp_files": True,
        },
    }

    progress_bar = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        "<",
        TimeRemainingColumn(),
        console=console,
    )

    with progress_bar:
        try:
            lesson = await parse_lesson_file(input_file)

            progress_adapter = RichProgressAdapter(progress_bar)

            # The factory will create the necessary services with their own defaults.
            processor = create_lesson_processor(
                tts_config=config["tts"],
                audio_config=config["audio"]
            )

            # Update the processor with our specific output directory.
            processor.output_dir = str(output_dir.absolute())

            result = await processor.process_lesson(
                lesson=lesson,
                output_dir=str(output_dir.absolute()),
                progress=progress_adapter
            )

            if not result or not result.get('success'):
                error_msg = result.get('error', 'Unknown error during processing.')
                print_error(f"Lesson processing failed: {error_msg}")
                raise typer.Exit(1)

            print_success("Lesson processing completed successfully!")
            if result.get('audio_file'):
                logger.info(f"Final audio file: {result['audio_file']}")

        except Exception as e:
            print_error(f"An unexpected error occurred: {e}")
            logger.exception("Lesson processing failed.")
            raise typer.Exit(1)

@app.command()
def generate(
    input_file: Path = typer.Argument(
        ...,
        help="Path to the lesson file to process.",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Base output directory for the generated files."
    ),
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose informational output.",
        ),
    ] = False,
):
    """
    Generates audio for a lesson file.
    """
    # Set logging level based on verbose flag
    log_level = logging.INFO if verbose else logging.WARNING
    logging.getLogger().setLevel(log_level)

    # Set default output directory if not provided.
    if output_dir is None:
        output_dir = Path.cwd() / "output"

    # Create a timestamped subdirectory for this run.
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = output_dir / f"run_{timestamp}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # Configure file logging for the run.
    log_file = run_output_dir / "tunatale.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG) # File logger is always DEBUG
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)

    asyncio.run(run_lesson_processing(input_file, run_output_dir))