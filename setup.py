import os
import setuptools


package_name = 'bdi-kit'
package_dir = 'bdikit'


def read_readme():
    with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf8') as file:
        return file.read()


def read_version():
    module_path = os.path.join(package_dir, '__init__.py')
    with open(module_path) as file:
        for line in file:
            parts = line.strip().split(' ')
            if parts and parts[0] == '__version__':
                return parts[-1].strip("'").strip("\"")

    raise KeyError('Version not found in {0}'.format(module_path))


def get_requires():
    with open('requirements.txt') as fp:
        dependencies = [line for line in fp if line and not line.startswith('#')]

        return dependencies


long_description = read_readme()
version = read_version()
requires = get_requires()
extra_requires = {}

setuptools.setup(
    name=package_name,
    version=version,
    packages=setuptools.find_packages(),
    install_requires=requires,
    extras_require=extra_requires,
    python_requires='>=3.9',
    description="bdi-kit library",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/VIDA-NYU/bdi-kit',
    include_package_data=True,
    author='',
    author_email='',
    maintainer='',
    maintainer_email='',
    keywords=['askem', 'data integration', 'nyu'],
    license='Apache-2.0',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering',
    ])