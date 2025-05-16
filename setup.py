from setuptools import setup, find_packages

setup(
    name="pyorps",  # Package name
    version="0.1.0",  # Version number
    author="Martin Hofmann",  # Author name
    author_email="martin.hofmann-3@ei.thm.de",  # Author email
    description="PYORPS (Python for Optimal Routes in Power Systems) is an open-source tool that automates "
                "underground cable route planning using high-resolution raster geodata. While tailored for "
                "distribution grids, it can be adapted for various infrastructures, optimizing routes for "
                "cost and environmental impact. ",  # Short description
    long_description=open("README.md").read(),  # Long description from README.md
    long_description_content_type="text/markdown",  # Content type for long description
    url="https://github.com/marhofmann/pyorps",  # Project homepage
    packages=find_packages(),  # Automatically find all packages in the project
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.10",  # Minimum Python version required
    install_requires=[
        "geopandas>=1.0.1",
        "numba>=0.61.2",
        "numpy>=2.2.5",
        "pandas>=2.2.3",
        "rasterio>=1.4.3",
        "networkit==11.1",
    ],
    extras_require={
        "viz": [
            "matplotlib>=3.10.1",
        ],
        "interactive": [
            "notebook>=7.4.0",
            "fiona>=1.10.1",
            "openpyxl>=3.1.5",
        ],
        "web": [
            "requests>=2.32.3",
        ],
        "graph": [
            "rustworkx>=0.16.0",
            "igraph>=0.11.8",
            "networkx>=3.4.2",
        ],
        "dev": [
            "coverage[toml]>=7.8.0",
            "pytest>=8.3.5",
        ],
        "full": [
            "matplotlib>=3.10.1",
            "notebook>=7.4.0",
            "fiona>=1.10.1",
            "openpyxl>=3.1.5",
            "requests>=2.32.3",
            "rustworkx>=0.16.0",
            "igraph>=0.11.8",
            "networkx>=3.4.2",
            "coverage[toml]>=7.8.0",
            "pytest>=8.3.5",
        ],
    },
    keywords=[
        "power line routing",
        "distribution grid",
        "power systems planning",
        "path finding",
        "rasterized GIS data",
    ],
    project_urls={
        "Homepage": "https://github.com/marhofmann/pyorps",
        "Bug Tracker": "https://github.com/marhofmann/pyorps/issues",
    },
    entry_points={
        "console_scripts": [
            "pyorps-cli=pyorps.cli:main",  # Replace with your CLI entry point if applicable
        ],
    },
)
