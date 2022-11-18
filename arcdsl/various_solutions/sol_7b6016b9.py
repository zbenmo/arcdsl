import numpy as np
from collections import Counter
from arcdsl.dsl import (
    Transform,
    Conditional,
    ConstMatrix,
    RepeatHorizontalPattern,
    create_foreach_cell,
    create_foreach_row,
    MaskByValue,
    ReplaceByMask
)
from arcdsl.utils import (
    connected_components,
    connected_component_to_bounding_rectangle
)
from arcdsl.solutions_registry import register


@register("7b6016b9")
def create_transform_7b6016b9(BACKGROUND=0, BLOCKED=2, FREE=3) -> Transform:
    get_background_mask = MaskByValue(BACKGROUND)

    def the_transformation(input_matrix: np.array) -> np.array:
        background_mask = get_background_mask(input_matrix)
        background_objects = connected_components(background_mask)
        blocked_mask = np.zeros(input_matrix.shape, dtype=np.uint8)
        free_mask = np.zeros(input_matrix.shape, dtype=np.uint8)
        for background in background_objects:
            background_bounding_rectangle = connected_component_to_bounding_rectangle(background)
            is_free = (
                background_bounding_rectangle[0].start == 0 or
                background_bounding_rectangle[1].start == 0 or
                background_bounding_rectangle[0].stop == input_matrix.shape[0] or
                background_bounding_rectangle[1].stop == input_matrix.shape[1]
            )
            for y, x in background:
                if is_free:
                    free_mask[y, x] = 1
                else:
                    blocked_mask[y, x] = 1

        add_free = ReplaceByMask(free_mask, FREE)
        add_blocked = ReplaceByMask(blocked_mask, BLOCKED)
        return add_blocked(add_free(input_matrix))

    return the_transformation
