"""Setup for pip package."""
import os
from sys import platform

from setuptools import Extension, dist, find_packages, setup


def readme():
    with open("README.md", "r") as f:
        return f.read()


class BinaryDistribution(dist.Distribution):
    """This class is needed in order to create OS specific wheels."""

    def has_ext_modules(self):
        return True


def get_version_number(default):
    # The `or default` is because on CI the `getenv` can return the empty string.
    version = os.getenv("LCE_RELEASE_VERSION", default) or default
    if "." not in version:
        raise ValueError(f"Invalid version: {version}")
    return version


ext_modules = [Extension("_foo", ["stub.cc"])] if platform.startswith("linux") else []

setup(
    name="larq-compute-engine",
    version=get_version_number(default="0.8.0"),
    python_requires=">=3.7",
    description="Highly optimized inference engine for binarized neural networks.",
    long_description=readme(),
    long_description_content_type="text/markdown",
    author="Plumerai",
    author_email="opensource@plumerai.com",
    packages=find_packages(),
    ext_modules=ext_modules,
    url="https://larq.dev/",
    install_requires=[
        "flatbuffers>=1.12,<2.0",
        "packaging>=19",
        "tqdm>=4",
        "importlib-metadata ~= 2.0 ; python_version<'3.8'",
    ],
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
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
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
