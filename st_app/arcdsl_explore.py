import streamlit as st

import pathlib
import numpy as np
# from itertools import islice
import inspect

from arcdsl.utils import (
    get_task,
    description_to_matrix,
    show_sample,
    examine_solution
)

from arcdsl.solutions_registry import registry


def examine_solution(train_task, solution_creation):
    st.header(train_task)
    st.code(inspect.getsource(solution_creation))
    file_contents = get_task(train_task)
    solution = solution_creation()
    for i, training_example in enumerate(file_contents['train']):
        st.subheader(f'{i}')
        minput, moutput = description_to_matrix(training_example['input']), description_to_matrix(training_example['output'])
        mpredicted = solution(minput) # good?
        st.write(f'{minput.shape} -> {moutput.shape}')
        if not np.array_equal(moutput, mpredicted):
            st.write("NOP!!!")
        st.pyplot(show_sample(minput, mpredicted, False))


training_path = pathlib.Path("training")

st.title('ARCDSL exploration')

tab1, tab2 = st.tabs(["All tasks", "Tasks with a solution"])

with tab1:

    train_task = st.selectbox("all training tasks", training_path.glob('*.json'))
    st.header(train_task)

    if train_task.name.replace('.json', '') in registry:
        st.write("Solution available!")

    file_contents = get_task(train_task)
    # print(file_contents)
    for i, training_example in enumerate(file_contents['train']):
        st.subheader(f'{i}')
        minput, moutput = description_to_matrix(training_example['input']), description_to_matrix(training_example['output'])
        st.write(f'{minput.shape} -> {moutput.shape}')
        st.pyplot(show_sample(minput, moutput, show=False))


with tab2:

    # for key, val in registry.items():
    #     examine_solution(training_path / f"{key}.json", val())

    selected_key_from_solutions_registry = st.selectbox("tasks with a solution", registry.keys())

    examine_solution(
        training_path / f"{selected_key_from_solutions_registry}.json",
        registry[selected_key_from_solutions_registry]
    )
