# arcdsl

A DSL for the ARC competition

https://www.kaggle.com/competitions/abstraction-and-reasoning-challenge/overview/timeline

Soon a new competition may start (ARC 2).

Also:

https://lab42.global/about-arc/ (active these days).

This competition is about AI in a very basic yet challenging way.

We're given tasks (inductive reasoning).
Each task is input -> output pairs, about 3-4 of those. Then another input and we're asked to generalize what are the rules that lead for the previous examples.
Inputs and outputs are given as a colorful matrix. Kind of a simple image.

A human can solve almost all tasks very easily.

How to achieve that with software? ML? DNN? RL? Search? Rule based?

While playing with ideas, I have decided to try to solve the tasks with manual written software.
So I've created a "Trasform" that receives an input matrix and produces the desired output.
This transformer can be tested on all train examples, and then used to provide an answer for the test/evaluation input.
There is some effort involved also here. While each task is clear once you figure out the logic, you still need to write some code. Not clear how easy it will be for a computer both to discover what needs to be done and then to create the mechanism to do it.

To help myself and maybe in the future also to help the computer, I'm creating a DSL to simplify the programming. The idea is that a higher building blocks, can reduce the effort. Also helps me to think about the challenge.
Most operations are wrappers around numpy. While the tasks are given in a JSON format, those are quickly turned into a numpy array, and then I do my best not to invent the wheel.

Also trying to write nice Pythonic code, with ideas from https://www.youtube.com/c/ArjanCodes and others. 

A 'Transform' is a protocol (actually a Callable). It receives a 2D matrix and returns a 2D matrix. The output matrix is not necessarily of the same size. Sometimes a transformer is simple and does not rely on the input matrix. This can be for example a horizontal flip. But when the transformation is dependant on the input matrix, or other stuff from the training / test examples, a wrapper or a closure may be needed.

IMPORTANT: get access to the (training) tasks from Kaggle.

You can try from a notebook (an example is provided), or also with streamlit. I've added the 'runstapp' script.

