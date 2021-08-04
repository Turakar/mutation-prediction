#!/bin/bash

black -q --check .
pyflakes .
isort -q --check .
