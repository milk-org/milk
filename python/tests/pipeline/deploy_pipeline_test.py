from __future__ import annotations

import pytest

from milk.errors import TaskDidNotSucceedError
from milk.pipeline import Pipeline
from milk.provision.rootdir import init
from milk.operate.tasks import Task


@pytest.fixture
def pipeline(pytestconfig, tmp_path):
    conf = pytestconfig.rootpath / "tests" / "resources" / "pipelinebasic-conf"
    return Pipeline(init(conf, parent=tmp_path))


class TestManualPipeline:

    def test_make_pipeline(self, pipeline):
        """
        Test that we can instantiate the pipeline directly in the source test folder
        """
        pp: Pipeline = pipeline
        assert pp.loop_number == 0
        assert pp.long_name == "pipelinebasic"

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

    def test_deploy_fps_call_by_class(self, cloned_pipeline: Pipeline):
        pp = cloned_pipeline

        from milk.operate.tasks import InitFPS

        pp.task_do(InitFPS)

        for cu, compu in pp.compute_units.items():
            assert compu.fps is not None
            assert compu.fps.is_valid()

            assert compu.fps.name == pp.config.fps_name(cu)

            compu.fps.destroy()

    def test_task_not_succeeding(self, cloned_pipeline: Pipeline):

        class Never(Task):

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

        from milk.operate.tasks import ConfStart, InitFPS

        pp.task_do(InitFPS)
        pp.task_do(ConfStart)

        live_names = set()
        for compu in pp.compute_units.values():
            assert compu.fps
            assert compu.fps.conf_isrunning()
            live_names.add(compu.fps.name)
        assert live_names == {"basic_delay0", "basic_delay1"}
