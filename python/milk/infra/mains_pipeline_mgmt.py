from __future__ import annotations
import typing as typ

from ..pipeline import Pipeline

from pathlib import Path

import os
import click

if typ.TYPE_CHECKING:
    from .task_models import SimpleTask

MILK_SAMPLES_DIR = os.environ["MILK_ROOT"] + "/milk-examples/pipelines/"
CACAO_SAMPLES_DIR = os.environ["MILK_ROOT"] + "/plugins/cacao-src/examples/"


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "-t",
    "--task",
    "task_arg",
    metavar="TASK1,TASK2,...",
    default=None,
    help="Comma-separated list of tasks to apply to the pipeline",
)
@click.option(
    "-c",
    "--configure",
    "configure_flag",
    is_flag=True,
    help='Shorthand for -t "configure"',
)
@click.option(
    "-r",
    "--run",
    "run_flag",
    is_flag=True,
    help='Shorthand for -t "run"',
)
@click.option(
    "--folder",
    "folder",
    type=click.Path(),
    default=None,
    help="Parent folder to the pipeline configuration (defaults to cwd)",
)
@click.option(
    "--cacao-exs",
    "cacao_exs",
    is_flag=True,
    help="Use a cacao example pipeline instead of --folder",
)
@click.option(
    "--milk-exs",
    "milk_exs",
    is_flag=True,
    help="Use a milk example pipeline instead of --folder",
)
@click.argument("_positional", nargs=-1)
def milk_pipes(
    task_arg: str | None,
    configure_flag: bool,
    run_flag: bool,
    folder: str | None,
    cacao_exs: bool,
    milk_exs: bool,
    _positional: tuple[str, ...],
):
    """
    Deploy and/or manage a milk/cacao pipeline.

    Usage:
      milk-pipes [-t task1,task2,...|-c|-r]
                 [--folder <pipefolder>|--cacao-exs|--milk-exs]
                 <pipename>

    Run `milk-pipes tasks` for a detailed list of tasks.
    """
    if len(_positional) == 0:
        print(milk_pipes.get_help(click.get_current_context()) + "\n")
        if milk_exs:
            browse_and_print_conf_folders(MILK_SAMPLES_DIR)
        if cacao_exs:
            browse_and_print_conf_folders(CACAO_SAMPLES_DIR)
        if folder is None:
            browse_and_print_conf_folders(os.getcwd())
        else:
            browse_and_print_conf_folders(folder)
        return

    return milk_pipes_function(
        task_arg, configure_flag, run_flag, folder, cacao_exs, milk_exs, _positional
    )


def milk_pipes_function(
    task_arg: str | None = None,
    configure_flag: bool = False,
    run_flag: bool = False,
    folder: str | None = None,
    cacao_exs: bool = False,
    milk_exs: bool = False,
    _positional: tuple[str, ...] = (),
):

    assert len(_positional) == 1  # temp
    (_positional_str,) = _positional  # unpack, since len is 1 for now.
    if _positional == "tasks":
        return milk_pipes_tasks_help()

    pipe_conf_name = _positional_str

    if sum([bool(task_arg), configure_flag, run_flag]) > 1:
        raise click.UsageError(
            "-t/--task, -c/--configure and -r/--run are mutually exclusive"
        )

    # TODO replace this with a task lookup relating to the pipeline
    tasks: list[type[SimpleTask]]
    if task_arg is not None:
        tasks = _parse_tasks([t.strip() for t in task_arg.split(",") if t.strip()])
    elif configure_flag:
        tasks = _get_tasks_configure()
    elif run_flag:
        tasks = _get_tasks_run()
    else:
        tasks = []  # TODO probably also an error

    if sum([bool(folder), milk_exs, cacao_exs]) > 1:
        raise click.UsageError(
            "--cacao-exs, --milk-exs and --folder are mutually exclusive"
        )

    if folder is not None:
        pipe_parent_folder = folder
    else:
        assert os.path.isdir(os.environ.get("MILK_ROOT", ""))
        if milk_exs:
            pipe_parent_folder = MILK_SAMPLES_DIR
        elif cacao_exs:  # cacao-exs
            pipe_parent_folder = CACAO_SAMPLES_DIR
        else:
            pipe_parent_folder = os.getcwd()

    pp = Pipeline(pipe_parent_folder, pipe_conf_name)  # Raise on missing folder

    if configure_flag:
        pp = pp.clone_to(os.getcwd())

    if tasks is None:
        print("No tasks to perform. Exiting.")
        return

    for Task in tasks:
        print(f"Executing task {Task.__name__}...")
        pp.task_do(Task)


def milk_pipes_tasks_help():
    print("""
        milk-pipes tasks

        Tasks that can be invoked with the
            milk-pipe [-t/--tasks task1,task2,...]
        syntax:
        """)


def _get_tasks_configure() -> list[type[SimpleTask]]:
    return _parse_tasks(["InitialFolderSetup"])


def _get_tasks_run() -> list[type[SimpleTask]]:
    return _parse_tasks(["DeployFPS", "StartConfProcesses", "LoadDataFiles"])


def _parse_tasks(task_names: list[str]) -> list[type[SimpleTask]]:
    from . import deploy_tasks as _d

    TASK_DB = {
        "InitialFolderSetup": _d.InitialFolderSetup,
        "LoadDataFiles": _d.LoadDataFiles,
        "DeployFPS": _d.DeployFPS,
        "StartConfProcesses": _d.StartConfProcesses,
    }
    return [TASK_DB[name] for name in task_names]


def browse_and_print_conf_folders(folder: str | Path):
    folder = Path(folder)
    if not os.path.isdir(folder):
        print(f"{folder} is not a directory.")
        return

    print(f"Valid configurations found in {folder}:")
    for sf in os.listdir(folder):
        subfolder = Path(sf)
        if (
            sf.endswith("-conf")
            and os.path.isdir(folder / subfolder)
            and os.path.isfile(folder / subfolder / "conf.toml")
        ):
            print(f" -    {subfolder}")
    print("---")


def deploy_pipeline_by_path(parent_folder: str | Path, long_name: str) -> Pipeline:
    """
    Deploy a pipeline from a fullname to the configuration
    """
    ...


def deploy_pipeline_from_examples(example_kind: str, long_name: str) -> Pipeline:
    """
    Deploy a pipeline from the examples

    Find the example folder, clone it to pwd (or to target), then deploy
    """
    ...
