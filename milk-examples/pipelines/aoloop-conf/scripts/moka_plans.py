"""aoloop's own tasks and plans: takedark with the light removed, then put back."""

import time
from dataclasses import replace

from moka.operate.plans import PLANS as BUILTIN, Step
from moka.operate.tasks import Task

FPS_NAME = "mlatsim_wfscam"  # the simulated camera, of the mlatsim pipeline
KEY = "fluxtotal"
SETTLE = 1.0  # [s] for the camera to output frames at the new flux


class _LightSim(Task):
    """The simulated light is the camera's fluxtotal. While it is off, its value is kept in
    work/lightsim.fluxtotal, so that a later run (even of another process) can put it back.
    """

    @property
    def _saved(self):
        return self.pipeline.work_folder / "lightsim.fluxtotal"

    def _fps(self):
        from pyMilk.interfacing.fps import FPS, FPSDoesntExistError

        try:
            return FPS(FPS_NAME)
        except FPSDoesntExistError:
            return None

    def _refusal(self) -> str | None:
        if self._fps() is None:
            return f"FPS {FPS_NAME} does not exist; deploy the mlatsim pipeline."
        return None


class RemoveLightSim(_LightSim):
    """Saves the camera's fluxtotal, then sets it to 0."""

    def _refusal(self) -> str | None:
        if reason := super()._refusal():
            return reason
        if not self._saved.is_file() and self._fps()[KEY] == 0:
            return f"{FPS_NAME}.{KEY} is already 0, with no saved value to put back."
        return None

    def forward(self) -> None:
        fps = self._fps()
        # A saved value is the one to put back: the light is already off, or half off.
        if not self._saved.is_file():
            self._saved.write_text(repr(float(fps[KEY])))
        fps[KEY] = 0.0
        time.sleep(SETTLE)

    def success(self) -> bool:
        fps = self._fps()
        return fps is not None and self._saved.is_file() and fps[KEY] == 0


class PutbackLightSim(_LightSim):
    """Sets the camera's fluxtotal back to the saved value; nothing to do without one."""

    def forward(self) -> None:
        self._fps()[KEY] = float(self._saved.read_text())
        self._saved.unlink()
        time.sleep(SETTLE)

    def success(self) -> bool:
        return not self._saved.is_file()


_takedark = BUILTIN["takedark"]

TASKS = [RemoveLightSim, PutbackLightSim]
PLANS = [
    replace(
        _takedark,
        steps=(Step("RemoveLightSim"), *_takedark.steps, Step("PutbackLightSim")),
    )
]
"""
The same plan, with every field of the built-in one passed again to the constructor
(needs: from moka.operate.plans import Plan):

PLANS = [
    Plan(
        _takedark.name,
        (Step("RemoveLightSim"), *_takedark.steps, Step("PutbackLightSim")),
        on_failure=_takedark.on_failure,
        accepts=_takedark.accepts,
        compus_arg=_takedark.compus_arg,
        params=_takedark.params,
        help=_takedark.help,
    )
]
"""
"""
The same plan again, fully restated without the built-in one
(needs: from moka.operate.plans import Param, Plan):

PLANS = [
    Plan(
        "takedark",
        (Step("RemoveLightSim"), Step("TakeDark"), Step("PutbackLightSim")),
        on_failure="abort",
        accepts="consistent",
        compus_arg=False,
        params=(
            Param("nframes", int, short="n", help="Average NFRAMES frames."),
            Param(
                "seconds",
                float,
                short="t",
                help="Average SECONDS of frames at mlat's framerate (5 s if neither -n nor -t).",
            ),
        ),
        help="Average wfsim into the WFS dark wfsdark.",
    )
]
"""
