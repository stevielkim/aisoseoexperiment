"""Setup script for geoseo_analysis package."""

from setuptools import setup, find_packages

# Read requirements from requirements.txt
# Only include essential runtime dependencies, not version-pinned ones
# (let the existing environment handle dependency management)
requirements = []

setup(
    name='geoseo_analysis',
    version='0.1.0',
    description='SEO vs AISO Analysis: Understanding what drives AI search engine citations',
    author='Stephanie Kim',
    author_email='stephaniekim@example.com',
    python_requires='>=3.9',
    packages=find_packages(where='.', exclude=['tests*', 'scripts*', 'notebooks*']),
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'jupyter>=1.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'geoseo-analyze=scripts.run_analysis:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
    project_urls={
        'Source': 'https://github.com/yourusername/geoseo_analysis',
        'Bug Reports': 'https://github.com/yourusername/geoseo_analysis/issues',
    },
)
