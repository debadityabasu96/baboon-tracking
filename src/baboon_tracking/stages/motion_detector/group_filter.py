import numpy as np
from typing import Tuple
from baboon_tracking.mixins.moving_foreground_mixin import MovingForegroundMixin
from baboon_tracking.models.frame import Frame
from pipeline import Stage
from pipeline.decorators import stage
from pipeline.stage_result import StageResult


@stage("moving_foreground")
class GroupFilter(Stage, MovingForegroundMixin):
    def __init__(self, moving_foreground: MovingForegroundMixin) -> None:
        Stage.__init__(self)
        MovingForegroundMixin.__init__(self)

        self._moving_foreground = moving_foreground

    def _pixel_has_neighbors(self, coord: Tuple[int, int]):
        x_coord, y_coord = coord
        moving_foreground = self._moving_foreground.moving_foreground.get_frame()

        height, width = moving_foreground.shape

        if moving_foreground[y_coord, x_coord] == 0:
            return False

        min_x = max(x_coord - 1, 0)
        max_x = min(x_coord + 1, width - 1)

        min_y = max(y_coord - 1, 0)
        max_y = min(y_coord + 1, height - 1)

        count = 0
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                if y == y_coord and x == x_coord:
                    continue

                if moving_foreground[y, x] > 0:
                    count += 1

                if count >= 7:
                    return True

        return False

    def execute(self) -> StageResult:
        moving_foreground = np.zeros_like(
            self._moving_foreground.moving_foreground.get_frame()
        )
        height, width = moving_foreground.shape

        for y in range(height):
            for x in range(width):
                if self._pixel_has_neighbors((x, y)):
                    moving_foreground[y, x] = 255

        self.moving_foreground = Frame(
            moving_foreground,
            self._moving_foreground.moving_foreground.get_frame_number(),
        )

        return StageResult(True, True)
