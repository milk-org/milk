from __future__ import annotations

import os
import pytest
from pathlib import Path

from milk.errors import (
    ConfigError,
    TaskCannotRunError,
    TaskDidNotSucceedError,
)
from milk.pipeline import Pipeline
from milk.infra.task_models import SimpleTask


@pytest.fixture
def pipeline(pytestconfig):
    return Pipeline(pytestconfig.rootpath / "tests" / "resources", "pipelinebasic")


class TestManualPipeline:

    def test_make_pipeline(self, pipeline):
        """
        Test that we can instantiate the pipeline directly in the source test folder
        """
        pp: Pipeline = pipeline
        assert pp.loop_number == 0
        assert pp.long_name == "pipelinebasic"

        assert pp.parent_folder.is_absolute()
        # Check the pipeline is the one directly from this repos' test sources
        # In the python/tests/resources folder
        assert pp.parent_folder.is_relative_to(Path(__file__).parent.parent)

    def test_pipeline_var_substitution(self, pipeline):
        pp: Pipeline = pipeline
        assert pp.loop_number == 0
        assert pp.long_name == "pipelinebasic"

        cus = pp.config.compute_units
        assert (
            cus["delay0"].fps["procinfo"]["triggersname"]
            == cus["delay0"].fps["in_name"]
        )
        assert (
            cus["delay1"].fps["procinfo"]["triggersname"]
            == cus["delay0"].fps["out_name"]
        )
        assert cus["delay1"].fps["in_name"] == cus["delay0"].fps["out_name"]

    def test_make_cloned_pipeline(self, cloned_pipeline):
        """
        Clone the pipeline to a working folder
        """
        pp: Pipeline = cloned_pipeline
        assert pp.loop_number == 0
        assert pp.long_name == "pipelinebasic"

        assert pp.parent_folder.is_absolute()
        assert pp.parent_folder.is_relative_to(os.getcwd())
        if ".nox" in pp.parent_folder.parts:
            assert "tmp" in pp.parent_folder.parts
        else:
            assert pp.parent_folder.is_relative_to("/tmp")

    def test_deploy_fps_call_by_class(self, cloned_pipeline: Pipeline):
        pp = cloned_pipeline

        from milk.infra.deploy_tasks import DeployFPS, InitialFolderSetup

        # Now forbidden because it MUST have a rootdir to perform a DeployFPS
        with pytest.raises(TaskCannotRunError) as cannot:
            pp.task_do(DeployFPS)
        assert cannot.value.task == "DeployFPS"
        assert cannot.value.pipeline == pp.short_name
        assert "does not exist" in str(cannot.value)  # the task's failure_reason
        pp.task_do(InitialFolderSetup).task_do(DeployFPS)

        for cu, compu in pp.compute_units.items():
            assert compu.fps is not None
            assert compu.fps.is_valid()

            assert compu.fps.name == pp.config.fps_name(cu)

            compu.fps.destroy()

    def test_task_not_succeeding(self, cloned_pipeline: Pipeline):

        class Never(SimpleTask):

            def can(self) -> bool:
                return True

            def success(self) -> bool:
                return False

            def forward(self) -> None: ...

        with pytest.raises(TaskDidNotSucceedError) as failed:
            cloned_pipeline.task_do(Never)
        assert failed.value.task == "Never"
        assert failed.value.pipeline == cloned_pipeline.short_name

    def test_deploy_fps_call_by_instance(self, cloned_pipeline: Pipeline):
        pp = cloned_pipeline

        from milk.infra.deploy_tasks import (
            DeployFPS,
            InitialFolderSetup,
            StartConfProcesses,
        )

        pp.task_do(InitialFolderSetup)
        pp.task_do(DeployFPS)
        pp.task_do(StartConfProcesses)

        live_names = set()
        for compu in pp.compute_units.values():
            assert compu.fps
            assert compu.fps.conf_isrunning()
            live_names.add(compu.fps.name)
        assert live_names == {"basic_delay0", "basic_delay1"}


from milk.infra import mains_pipeline_mgmt as mgmt


class TestMilkPipes:

    def test_no_pipelines(self):
        # No folder
        with pytest.raises(ConfigError):
            mgmt.milk_pipes_init_function("yolo")
        # No conf.toml
        os.mkdir("yolo-conf")
        with pytest.raises(ConfigError):
            mgmt.milk_pipes_init_function("yolo")
