from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tts-processor",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Multi-voice TTS processor for language learning scripts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/tts-processor",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "edge-tts>=6.1.0",
        "pydub>=0.25.1",
    ],
    entry_points={
        'console_scripts': [
            'tts-processor=tts_processor.cli:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
