from setuptools import setup, find_packages

setup(
    name='kobertscore',
    version='0.0.1',
    description='KoBERT score for super-natural dataset and unnatural dataset',
    author='sylee96',
    author_email='syoon369@gmail.com',
    url='',
    install_requires=[],
    packages=find_packages(exclude=[]),
    keywords=['kobert_score,', 'compare_similarity'],
    python_requires='>=3.6',
    package_data={},
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)