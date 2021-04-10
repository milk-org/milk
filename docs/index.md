**Contents**
* TOC
{:toc}

***


# What is milk ?

milk provides a framework and tools for image processing and analysis. milk is geared towards high-performance computing and modularity, and is especially well-suited for building up execution pipelines out of small modular units. The framework allows for custom functions to be easily integrated within milk's command line interface.

milk's key features include :

- A command line interface (CLI) for easy access to functions and arguments
- Runtime loading of modules to extend capabilities
- Interprocess communication (IPC) for sharing process status, function parameters, and images.

milk's core design and capabilities are built around three data structures :

- [Shared memory images (streams)](streams.md)
- [Function Parameter Structures (FPS)](fps.md)
- [Process info and status (procinfo)](procinfo.md)







# Installing milk

    ./compile.sh $PWD/local

# Adding plugins

Compile with cacao plugins:

    ./fetch_cacao_dev.sh
    ./compile.sh $PWD/local

Compile with coffee plugins:

    ./fetch_coffee_dev.sh
    ./compile.sh $PWD/local


# Contributing







For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).
