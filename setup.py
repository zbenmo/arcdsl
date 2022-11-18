from setuptools import setup, find_packages

setup(
    name='arcdsl',
    maintainer='zbenmo@gmail.com',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.23.4, <1.24',
        'opencv-contrib-python>=4.6.0.66, <4.7',
    ],
    extras_require={
        'interactive': [
            'matplotlib>=3.6.1, <3.7',
            'jupyterlab>=3.4.8, <3.5',
            'streamlit>=1.15.0, <1.16',
        ],
    },
    scripts=['bin/runstapp'],
)