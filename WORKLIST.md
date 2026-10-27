# OTHER WORKLIST

- Add a test for the ImageStreamIO version, got burned AGAIN.

# Worklist for the deployment package.

This file describes project goals for the development of the python interface to PIPELINES

A pipeline is a set of compute units, under a single configuration set and loop number

This pipeline layer will define how configurations are loaded from toml files, which may recursively load each other.

I'm working my way from the [cacao-docs](https://cacao-org.github.io/docs/) and establishing what should be done.

All the work should be done in the python/milk python package.

## Begin specifications

```
cacao-loop-deploy — which does BOTH deployment and example fetcher
cacao-setup
```
```
register CU and cacao-fpslistadd
```

I need arbitrary MILK pipelines deployer. Typ. a graph-gen to check them
And then some "loadable defaults" which MAY use variables that are yet to be defined, like templates (e.g. the loopNumber should be templatable after loading the default)

Env variable toml override


## What is "a folder" and how does a pipeline operate.
