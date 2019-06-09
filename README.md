[![Build status](https://img.shields.io/appveyor/ci/olegtarasov/fasttext-netwrapper.svg?logo=appveyor)](https://ci.appveyor.com/project/olegtarasov/fasttext-netwrapper/branch/master)
[![Nuget](https://img.shields.io/nuget/v/FastText.NetWrapper.svg?logo=nuget)](https://www.nuget.org/packages/FastText.NetWrapper)
![Supported platforms](https://img.shields.io/badge/platforms-Windows%2C%20Linux%2C%20MacOs-blue.svg)

# FastText.NetWrapper

This is a cross-platform .NET Standard wrapper for Facebook's [FastText](https://github.com/facebookresearch/fastText) library. The wrapper comes with bundled precompiled native binaries for all three platforms: Windows, Linux and MacOs.

Just add it to your project and start using it! No additional setup required. This library will unpack and call appropriate native binary depending on target platform.

## Requirements

Since this wrapper uses native C++ binaries under the hood, you will need to have Visual C++ Runtime Version 140 installed. Visit the MS Downloads page (https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads) and select the appropriate redistributable. 

## FastText C-style API

If you are interested in using FastText with C-style API, here is my fork of the official library: https://github.com/olegtarasov/fastText.