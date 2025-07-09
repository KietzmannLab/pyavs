"""
Setup script for pyAVS package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Python package for Active Visual Semantics dataset processing"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    # Default requirements if file doesn't exist
    return [
        'numpy>=1.19.0',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        'matplotlib>=3.3.0',
        'h5py>=3.1.0',
        'pillow>=8.0.0',
        'scikit-image>=0.18.0',
        'pycocotools>=2.0.0',
        'mne>=1.0.0',
    ]

setup(
    name='pyavs',
    version='0.1.0',
    author='P. Sulewski, C. Meinert',
    author_email='psulewski@uos.de',
    description='Python package for Active Visual Semantics dataset processing',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/your-org/pyavs',  # Update with actual repository URL
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'flake8>=3.8',
            'isort>=5.0',
            'sphinx>=4.0',
            'sphinx-rtd-theme>=1.0',
        ],
        'full': [
            'autoreject>=0.3.0',
            'sklearn>=1.0.0',
            'seaborn>=0.11.0',
            'jupyter>=1.0.0',
        ],
    },
    package_data={
        'pyavs': [
            'examples/*.py',
            'data/*.csv',
        ],
    },
    entry_points={
        'console_scripts': [
            'pyavs=pyavs.cli:main',
        ],
    },
    keywords=[
        'neuroscience',
        'MEG',
        'eye-tracking',
        'visual-semantics',
        'BIDS',
        'neuroimaging',
        'cognitive-science',
    ],
    project_urls={
        'Bug Reports': 'https://github.com/your-org/pyavs/issues',
        'Source': 'https://github.com/your-org/pyavs',
        'Documentation': 'https://pyavs.readthedocs.io/',
    },
)