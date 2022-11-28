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


def examine_solution(train_task, solutions_creation):
    st.header(train_task)
    for solution_creation in solutions_creation:
        st.code(inspect.getsource(solution_creation))
    if len(solutions_creation) > 1:
        st.write('Using the last solution (above this text line) in next examinations.')
    solution_creation = solutions_creation[-1]
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
    for i, test_example in enumerate(file_contents['test']):
        st.subheader(f'test')
        minput = description_to_matrix(test_example['input'])
        mpredicted = solution(minput) # good?
        st.write(f'{minput.shape} -> {mpredicted.shape}')
        # if not np.array_equal(moutput, mpredicted):
        #     st.write("NOP!!!")
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
    for i, test_example in enumerate(file_contents['test']):
        st.subheader(f'test')
        minput = description_to_matrix(test_example['input'])
        st.write(f'{minput.shape} -> ?')
        st.pyplot(show_sample(minput, np.zeros_like(minput), show=False))


with tab2:

    # for key, val in registry.items():
    #     examine_solution(training_path / f"{key}.json", val())

    selected_key_from_solutions_registry = st.selectbox("tasks with a solution", registry.keys())

    examine_solution(
        training_path / f"{selected_key_from_solutions_registry}.json",
        registry[selected_key_from_solutions_registry]
    )
