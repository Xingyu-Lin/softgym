from setuptools import setup

requirements = [
    'autolab_core',
    'gym',  
    'gputil'
]

setup(name='flex-gym',
      version='0.0.1',
      description='Flex Gym',
      author='NVIDIA Flex',
      author_email='',
      package_dir = {'': 'src'},
      packages=['flex_gym'],
      install_requires=requirements,
     )
