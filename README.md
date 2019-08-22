# MLS-MPM

This is a more readable implementation of MLS-MPM, originally written in 88 LOC
by Yuanming Hu to implement the method described in "A Moving Least Squares
Material Point Method with Displacement Discontinuity and Two-Way Rigid Body
Coupling" from SIGGRAPH 2018.

The readable version can be found in `mls-mpm.cpp` while the original is in
`orig-mls-mpm88.cpp`. In March 2019, Hu posted another readable alternative,
which can be found in `orig-mls-mpm88-explained.cpp`.

## How to Install Dependencies (Ubuntu)

    sudo apt install cmake build-essential

    # optional
    sudo apt install libgomp1

## How to Build

    cmake -H. -Bbuild
    cmake --build build

## How to Run

    build/mls-mpm
