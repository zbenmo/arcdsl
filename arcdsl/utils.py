import json
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2


def get_task(file_path):
    return json.load(open(file_path))


def description_to_matrix(description: List) -> np.array:
    return np.array(description, dtype=np.uint8)


def show_sample(minput, moutput):
    fig = plt.figure(constrained_layout=True)

    gs = GridSpec(1, 2, figure=fig)
    ax = [fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1])]

    for i, data in enumerate((minput, moutput)):
        ax[i].imshow(data, vmin=0, vmax=7)

        # the text
        for row_i, row in enumerate(data):
            for col_i, cell in enumerate(row):
                ax[i].text(col_i, row_i, cell, va='center', ha='center')

    # making the sizes correct to match for example (3x3) -> (9x9) etc.
    max_shape_y = max(minput.shape[0], moutput.shape[0])
    max_shape_x = max(minput.shape[1], moutput.shape[1])

    for i, data in enumerate((minput, moutput)):
        ax[i].set_axis_off()

        half_diffy = (max_shape_y - data.shape[0]) / 2
        half_diffx = (max_shape_x - data.shape[1]) / 2

        # using here ax[1] as '1' is the (indmax!!!): TODO!!
        limy = [lim - half_diffy for lim in ax[1].get_ylim()]
        limx = [lim - half_diffx for lim in ax[1].get_xlim()]

        ax[i].set_xlim(limx)
        ax[i].set_ylim(limy)

    # see https://stackoverflow.com/a/66977230/1614089        

    plt.show()


def examine_solution(train_task, solution):
    file_contents = get_task(train_task)
    print(train_task)
    # print(file_contents)
    for i, training_example in enumerate(file_contents['train']):
        print(f'>> {i}')
        minput, moutput = description_to_matrix(training_example['input']), description_to_matrix(training_example['output'])
        mpredicted = solution(minput) # good?
        print(f'{minput.shape} -> {moutput.shape}')
        if not np.array_equal(moutput, mpredicted):
            print("NOP!!!")
        show_sample(minput, mpredicted)


def connected_components(input_matrix: np.array):
    """
    Returns a list, where each item is a connected component. The items are given as a list of the relevant cells.
    Assumes the first label is the background and hense it is skipped.
    """

    nlabels, labels = cv2.connectedComponents(input_matrix, connectivity=4)
    x_ys = [
        np.where(labels == i)
        for i in range(1, nlabels)
    ]
    return [list(zip(*x_y)) for x_y in x_ys]


def connected_component_to_bounding_rectangle(connected_component: list) -> np.index_exp:
    """
    Returns the bounding rectange of the component, in terms of min_x, max_x, min_y, max_y
    """

    ys, xs = list(zip(*connected_component))
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return np.index_exp[min_y:max_y + 1, min_x:max_x + 1]
