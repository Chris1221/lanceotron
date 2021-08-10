from setuptools import setup

setup(
   name='lanceotron',
   version='0.1',
   python_requires='>3.6', 
   description='Lanceotron',
   author='Chris Cole',
   author_email='ccole@well.ox.ac.uk',
   packages=['lanceotron'],  #same as name
   entry_points={
        'console_scripts': [
            'lanceotron = lanceotron.cli:genome',
        ],
    }
)