[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/olegtarasov/FastText.NetWrapper/Build%20and%20publish?style=flat-square)](https://github.com/olegtarasov/FastText.NetWrapper/actions)
[![Nuget](https://img.shields.io/nuget/v/FastText.NetWrapper?style=flat-square)](https://www.nuget.org/packages/FastText.NetWrapper)
[![Donwloads](https://img.shields.io/nuget/dt/FastText.NetWrapper?label=Nuget&style=flat-square)](https://www.nuget.org/packages/FastText.NetWrapper)

# FastText.NetWrapper

This is a cross-platform .NET Standard wrapper for Facebook's [FastText](https://github.com/facebookresearch/fastText) library. The wrapper comes with bundled precompiled native binaries for all three platforms: Windows, Linux and MacOs.

Just add it to your project and start using it! No additional setup required. This library will unpack and call appropriate native binary depending on target platform.

## Windows Requirements

Since this wrapper uses native C++ binaries under the hood, you will need to have Visual C++ Runtime Version 140 installed when running under Windows. Visit the MS Downloads page (https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads) and select the appropriate redistributable. 

## FastText C-style API

If you are interested in using FastText with C-style API, here is my fork of the official library: https://github.com/olegtarasov/fastText.