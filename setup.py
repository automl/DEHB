# -*- encoding: utf-8 -*-
import setuptools


def read_file(file_name):
    with open(file_name, encoding='utf-8') as fh:
        text = fh.read()
    return text


setuptools.setup(
    name='dehb',
    author_email='{awad, mallik, fh}@cs.uni-freiburg.de',
    description='Evolutionary Hyberband for Scalable, Robust and Efficient '
                'Hyperparameter Optimization',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    license='Apache-2.0',
    url='https://www.automl.org/automl/',
    project_urls={
        'Documentation': 'https://github.com/automl/dehb',
        'Source Code': 'https://github.com/automl/dehb'
    },
    version=read_file('dehb/__version__.py').split()[-1].strip('\''),
    packages=setuptools.find_packages(exclude=['*.tests', '*.tests.*',
                                               'tests.*', 'tests'],),
    python_requires='>3.5, <=3.9',
    install_requires=read_file('./requirements.txt').split('\n'),
    # extras_require=get_extra_requirements(),  # We dont need this currently.
    test_suite='pytest',
    platforms=['Linux'],
    classifiers=[
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
    ]
)
