from setuptools import setup

setup(
    name='arcdsl',
    maintainer='zbenmo@gmail.com',
    version='0.1.0',
    install_requires=[
        'numpy>=1.23.4, <1.24',
        'opencv-contrib-python>=4.6.0.66, <4.7',
    ],
    extras_require={
        'interactive': [
            'matplotlib>=3.6.1, <3.7',
            'jupyterlab>=3.4.8, <3.5',
        ],
    }
)