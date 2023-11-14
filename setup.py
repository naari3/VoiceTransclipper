from setuptools import setup, find_packages
with open('requirements.lock') as requirements_file:
    install_requirements = []
    # ignore comment lines and empty lines
    for line in requirements_file:
        if line.startswith('#') or line == '\n':
            continue
        install_requirements.append(line.strip())
setup(
    name="voicetransclipper",
    version="0.0.1",
    description="A small package",
    author="naari3",
    packages=find_packages(),
    install_requires=install_requirements,
    entry_points={
        "console_scripts": [
            "voicetransclipper=voicetransclipper.core:main",
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3.11',
    ]
)

