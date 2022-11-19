import numpy as np
from collections import Counter
from dataclasses import dataclass
from arcdsl.dsl import (
    Transform,
    Conditional,
    ConstMatrix,
    RepeatHorizontalPattern,
    create_foreach_cell,
    create_foreach_row,
    MaskByValue,
    ReplaceByMask,
    ReplaceSpecificValue
)
from arcdsl.utils import (
    connected_components,
    connected_component_to_bounding_rectangle
)
from arcdsl.solutions_registry import register


@register("007bbfb7")
def create_transform_007bbfb7() -> Transform:
    def the_transformation(input_matrix: np.array) -> np.array:
        conditional = Conditional([
            (lambda m: m == 0, ConstMatrix(np.zeros(input_matrix.shape, dtype=np.uint8))),
            (lambda _: True, ConstMatrix(input_matrix)),
        ])
        return create_foreach_cell(conditional)(input_matrix)
    return the_transformation


@register("25d8a9c8")
def create_transform_25d8a9c8() -> Transform:
    def the_transformation(input_matrix: np.array) -> np.array:
        conditional = Conditional([
            (lambda row: np.all(row == row[0, 0]), ConstMatrix(np.full((1, input_matrix.shape[1]), 5, dtype=np.uint8))),
            (lambda _: True, ConstMatrix(np.zeros((1, input_matrix.shape[1]), dtype=np.uint8))),
        ])
        return create_foreach_row(conditional, input_matrix.shape[1])(input_matrix)
    return the_transformation


@register("963e52fc")
def create_transform_963e52fc() -> Transform:
    """

    >>> create_transform_963e52fc()(np.array([[1, 0, 1], [0, 1, 0]]))
    array([[1, 0, 1, 0, 1, 0],
           [0, 1, 0, 1, 0, 1]])
    """

    def the_transformation(input_matrix: np.array) -> np.array:
        return RepeatHorizontalPattern(input_matrix.shape[1])(input_matrix)
    return the_transformation


@register("6855a6e4")
def create_transform_6855a6e4(MOVING=5, GATE=2, CLEAR=0) -> Transform:
    get_moving_mask = MaskByValue(MOVING)
    get_gate_mask = MaskByValue(GATE)

    def observe_gate(g: list) -> dict:
        ys, xs = list(zip(*g))
        x_counter = Counter(xs)
        y_counter = Counter(ys)
        vertical = len(x_counter) == 2
        horizontal = len(y_counter) == 2
        assert vertical ^ horizontal

        ret = {
            'vertical': vertical,
        }
        
        if vertical:
            x_body, x_count = x_counter.most_common()[0]
            x_add, x_add_count  = x_counter.most_common()[1]
            x_add = x_add - x_body
            assert x_add_count == 2
            ret.update({
                'x_body': x_body,
                # 'y_min': min(y_counter),
                # 'y_max': max(y_counter)
            })
        else:
            y_body, y_count  = y_counter.most_common()[0]
            y_add, y_add_count = y_counter.most_common()[1]
            y_add = y_add - y_body
            assert y_add_count == 2
            ret.update({
                'y_body': y_body,
                # 'x_min': min(x_counter),
                # 'x_max': max(x_counter)
            })
            
        return ret

    def distance(moving, gate):
        if gate['vertical']:
            distance = min(abs(cell[1] - gate['x_body']) for cell in moving)
        else:
            distance = min(abs(cell[0] - gate['y_body']) for cell in moving)
        return distance

    def find_closest(moving, gp: list) -> dict:
        
        # TODO: ? verify row in vertical and col in horozontal?
        return min([(distance(moving, gate), gate) for gate in gp])[1]


    def flip(v, axis):
        return v + 2 * (axis - v)


    def the_transformation(input_matrix: np.array) -> np.array:
        moving_mask = get_moving_mask(input_matrix)
        gate_mask = get_gate_mask(input_matrix)

        moving_objects = connected_components(moving_mask)
        gate_objects = connected_components(gate_mask)
        gates_processed = [observe_gate(g) for g in gate_objects]

        new_mask = np.zeros(input_matrix.shape, dtype=np.uint8)
        for moving in moving_objects:
            gate = find_closest(moving, gates_processed)
            ys, xs = list(zip(*moving))
            if gate['vertical']:
                new_mask[ys, [flip(x, gate['x_body']) for x in xs]] = 1
            else:
                new_mask[[flip(y, gate['y_body']) for y in ys], xs] = 1

        clear_moving_from_old_position = ReplaceByMask(moving_mask, CLEAR)
        add_moving_in_new_position = ReplaceByMask(new_mask, MOVING)  
        return add_moving_in_new_position(clear_moving_from_old_position(input_matrix))

    return the_transformation


@register("6d75e8bb")
def create_transform_6d75e8bb(SCAFFOLD=8, COMPLETE=2, CLEAR=0) -> Transform:
    get_scaffold_mask = MaskByValue(SCAFFOLD)

    def the_transformation(input_matrix: np.array) -> np.array:
        scaffold_mask = get_scaffold_mask(input_matrix)
        scaffold_objects = connected_components(scaffold_mask)
        complete_mask = np.zeros(input_matrix.shape, dtype=np.uint8)
        for scaffold in scaffold_objects:
            scaffold_bounding_rectangle = connected_component_to_bounding_rectangle(scaffold)
            complete_mask[scaffold_bounding_rectangle] = input_matrix[scaffold_bounding_rectangle] == CLEAR
        add_complete = ReplaceByMask(complete_mask, COMPLETE)
        return add_complete(input_matrix)

    return the_transformation


@register("6d75e8bb")
def alternative_ct_to_6d75e8bb(SCAFFOLD=8, COMPLETE=2, CLEAR=0) -> Transform:
    get_scaffold_mask = MaskByValue(SCAFFOLD)
    replace_specific_value = ReplaceSpecificValue(CLEAR, COMPLETE)

    def the_transformation(input_matrix: np.array) -> np.array:
        scaffold_mask = get_scaffold_mask(input_matrix)
        scaffold_objects = connected_components(scaffold_mask)
        output_matrix = input_matrix.copy()
        for scaffold in scaffold_objects:
            scaffold_bounding_rectangle = connected_component_to_bounding_rectangle(scaffold)
            output_matrix[scaffold_bounding_rectangle] = (
                replace_specific_value(input_matrix[scaffold_bounding_rectangle])
            )
        return output_matrix

    return the_transformation


@register("44d8ac46")
def create_transform_44d8ac46(SCAFFOLD=5, COMPLETE=2, CLEAR=0) -> Transform:
    get_scaffold_mask = MaskByValue(SCAFFOLD)
    get_clear_mask = MaskByValue(CLEAR)

    def the_transformation(input_matrix: np.array) -> np.array:
        scaffold_mask = get_scaffold_mask(input_matrix)
        scaffold_objects = connected_components(scaffold_mask)
        clear_mask = get_clear_mask(input_matrix)
        complete_mask = np.zeros(input_matrix.shape, dtype=np.uint8)
        for scaffold in scaffold_objects:
            scaffold_bounding_rectangle = connected_component_to_bounding_rectangle(scaffold)
            clear_objects = connected_components(clear_mask[scaffold_bounding_rectangle])
            for clear in clear_objects:
                clear_bounding_rectangle = connected_component_to_bounding_rectangle(clear)
                if (
                    clear_bounding_rectangle[0].stop - clear_bounding_rectangle[0].start !=
                    clear_bounding_rectangle[1].stop - clear_bounding_rectangle[1].start
                ) :
                    continue # apparently not a square
                min_y = scaffold_bounding_rectangle[0].start
                min_x = scaffold_bounding_rectangle[1].start
                adjusted_clear_ind_exp = np.index_exp[
                    min_y + clear_bounding_rectangle[0].start:min_y + clear_bounding_rectangle[0].stop,
                    min_x + clear_bounding_rectangle[1].start:min_x + clear_bounding_rectangle[1].stop,
                ]
                if np.all(input_matrix[adjusted_clear_ind_exp] == CLEAR): # are all cells clear?
                    complete_mask[adjusted_clear_ind_exp] = 1
        add_complete = ReplaceByMask(complete_mask, COMPLETE)
        return add_complete(input_matrix)

    return the_transformation


@register("44d8ac46")
def alternative_ct_to_44d8ac46(SCAFFOLD=5, COMPLETE=2, CLEAR=0) -> Transform:

    class Helper:
        get_clear_mask = MaskByValue(CLEAR)

        def __call__(self, input_matrix: np.array) -> np.array:
            output_matrix = input_matrix.copy()
            clear_mask = self.get_clear_mask(input_matrix)
            clear_objects = connected_components(clear_mask)
            for clear in clear_objects:
                clear_bounding_rectangle = connected_component_to_bounding_rectangle(clear)
                if (
                    clear_bounding_rectangle[0].stop - clear_bounding_rectangle[0].start !=
                    clear_bounding_rectangle[1].stop - clear_bounding_rectangle[1].start
                ) :
                    continue # apparently not a square
                if np.all(input_matrix[clear_bounding_rectangle] == CLEAR): # are all cells clear?
                    output_matrix[clear_bounding_rectangle] = COMPLETE
            return output_matrix

    get_scaffold_mask = MaskByValue(SCAFFOLD)
    helper = Helper()

    def the_transformation(input_matrix: np.array) -> np.array:
        output_matrix = input_matrix.copy()
        scaffold_mask = get_scaffold_mask(input_matrix)
        scaffold_objects = connected_components(scaffold_mask)
        for scaffold in scaffold_objects:
            scaffold_bounding_rectangle = connected_component_to_bounding_rectangle(scaffold)
            output_matrix[scaffold_bounding_rectangle] = (
                helper(input_matrix[scaffold_bounding_rectangle])
            )
        return output_matrix

    return the_transformation


if __name__ == "__main__":
    import doctest
    doctest.testmod()