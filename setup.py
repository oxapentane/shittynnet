import setuptools

with open("README.md", 'r') as f:
  long_description=f.read()

setuptools.setup(
  name="shittynnet",
  version="0.1.0",
  author="Grisha Shipunov",
  author_email="blame@oxapentane.com",
  description="Small and shitty autograd wirtten for education purposes",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/oxapentane/shittynnet",
  packages=setuptools.find_packages(),
  install_requires=['numpy'],
)

