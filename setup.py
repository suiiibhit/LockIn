from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lockin",
    version="1.0.0",
    author="Shubhit Saxena",
    author_email="shubhitsaxena2005@gmail.com",
    description="An anti-distraction productivity tool that detects phone usage via webcam and alerts you in real-time",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/suhibhit/lockin",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Video",
        "Topic :: Office/Business",
        "Development Status :: 4 - Beta",
    ],
    python_requires=">=3.9",
    install_requires=[
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0",
        "ultralytics>=8.0.0",
        "pygame>=2.5.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "optional": [
            "pydub>=0.25.0",
            "pyttsx3>=2.90",
        ],
    },
    entry_points={
        "console_scripts": [
            "lockin=lockin.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "lockin": ["sounds/*.wav", "config/*.json"],
    },
)
