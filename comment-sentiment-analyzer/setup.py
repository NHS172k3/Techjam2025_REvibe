from setuptools import setup, find_packages

setup(
    name='comment-sentiment-analyzer',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A classifier to evaluate video quality based on comments.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/comment-sentiment-analyzer',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'sentence-transformers',
        'torch',
        'transformers',
        'matplotlib',
        'seaborn',
        'PyYAML',
        'jupyter',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)