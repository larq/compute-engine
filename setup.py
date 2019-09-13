"""Setup for pip package."""

from setuptools import Extension, dist, find_packages, setup


class BinaryDistribution(dist.Distribution):
    """This class is needed in order to create OS specific wheels."""

    def has_ext_modules(self):
        return True


setup(
    name="larq-compute-engine",
    version="0.0.1",
    python_requires=">=3.4",
    description="An Open Source Collection of Highly Tuned Implementations of Primitives Operations for Binarized Neural Networks",
    author="Plumerai",
    author_email="arash@plumerai.com",
    packages=find_packages(),
    install_requires=[],
    extras_require={
        "tensorflow": ["tensorflow>=1.13.1"],
        "tensorflow_gpu": ["tensorflow-gpu>=1.13.1"],
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
