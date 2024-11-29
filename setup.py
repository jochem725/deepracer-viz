import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
  name='deepracer-viz',
  version='0.1',
  author='Jochem Lugtenburg',
  author_email='X',
  description="Visualizations for AWS DeepRacer videos",
  long_description=long_description,
  long_description_content_type='text/markdown',
  url='https://github.com/jochem725/deepracer-viz',
  packages=setuptools.find_packages(),
  install_requires=['tensorflow'],
  keywords="DeepRacer Visualization Python Tensorflow",
  classifiers=[
    'Programming Language :: Python :: 3',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Libraries',
    'Topic :: System :: Hardware',
  ],
  python_requires='>=3.8',
)
