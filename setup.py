import re
import setuptools
from collections import defaultdict

package_name = 'bdi'
package_dir = 'bdi'


def read_readme():
    with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf8') as file:
        return file.read()


def read_version():
    module_path = os.path.join(package_dir, '__init__.py')
    with open(module_path) as file:
        for line in file:
            parts = line.strip().split(' ')
            if parts and parts[0] == '__version__':
                return parts[-1].strip("'")

    raise KeyError('Version not found in {0}'.format(module_path))


def get_requires():
    with open('requirements.txt') as fp:
        dependencies = [line for line in fp if line and not line.startswith('#')]

        return dependencies


long_description = read_readme()
version = read_version()
requires = get_requires()
extra_requires = get_extra_requires()

setuptools.setup(
    name=package_name,
    version=version,
    packages=setuptools.find_packages(),
    install_requires=requires,
    extras_require=extra_requires,
    description="Mapper System",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/VIDA-NYU/alpha-automl',
    include_package_data=True,
    author='',
    author_email='',
    maintainer='',
    maintainer_email='',
    keywords=['askem', 'table mapping', 'nyu'],
    license='Apache-2.0',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering',
    ])