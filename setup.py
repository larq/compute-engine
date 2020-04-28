"""Setup for pip package."""

from setuptools import dist, Extension, find_packages, setup
from sys import platform


def readme():
    with open("README.md", "r") as f:
        return f.read()


class BinaryDistribution(dist.Distribution):
    """This class is needed in order to create OS specific wheels."""

    def has_ext_modules(self):
        return True


ext_modules = [Extension("_foo", ["stub.cc"])] if platform.startswith("linux") else []

setup(
    name="larq-compute-engine",
    version="0.2.1",
    python_requires=">=3.6",
    description="Highly optimized inference engine for binarized neural networks.",
    long_description=readme(),
    long_description_content_type="text/markdown",
    author="Plumerai",
    author_email="arash@plumerai.com",
    packages=find_packages(),
    ext_modules=ext_modules,
    url="https://larq.dev/",
    install_requires=["packaging>=19"],
    extras_require={
        "tensorflow": ["tensorflow>=1.14"],
        "tensorflow_gpu": ["tensorflow-gpu>=1.14"],
    },
    include_package_data=True,
    zip_safe=False,
    distclass=BinaryDistribution,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    license="Apache 2.0",
    keywords="binarized neural networks",
)
