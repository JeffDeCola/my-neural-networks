# PERCEPTRON SIMPLE EXAMPLE

[![jeffdecola.com](https://img.shields.io/badge/website-jeffdecola.com-blue)](https://jeffdecola.com)
[![MIT License](https://img.shields.io/:license-mit-blue.svg)](https://jeffdecola.mit-license.org)

_A simple an implementation of a perceptron (P) neural network
written in go._

Table of Contents

* [OVERVIEW](https://github.com/JeffDeCola/my-neural-networks/tree/main/perceptron/perceptron-simple-example#overview)

Documentation and Reference

* [artificial intelligence overview](https://github.com/JeffDeCola/my-cheat-sheets/tree/master/software/development/software-architectures/artificial-intelligence/ai-fundamentals/artificial-intelligence-overview-cheat-sheet#artificial-intelligence-overview-cheat-sheet)
* [neural networks](https://github.com/JeffDeCola/my-cheat-sheets/tree/master/software/development/software-architectures/artificial-intelligence/ai-fundamentals/neural-networks-cheat-sheet#neural-networks-cheat-sheet)
  * [mathematical model of a neuron](https://github.com/JeffDeCola/my-cheat-sheets/tree/master/software/development/software-architectures/artificial-intelligence/ai-fundamentals/neural-networks-cheat-sheet#mathematical-model-of-a-neuron)

## OVERVIEW

A perceptron is a simple single-layer neural network that takes N inputs,
applies a learned weight to each, sums them with a bias,
and produces a single output of 1 or 0 via a step activation function.

This is an illustration of a 2 input perceptron.

<p align="center">
    <img src="https://raw.githubusercontent.com/JeffDeCola/my-cheat-sheets/refs/heads/master/docs/pics/software/development/neural-networks-perceptron.svg"
    alt="perceptron">
</p>

## MODEL A NEURON IN GO

First, we must model a neuron in go. From my cheat sheet, we made a model of a neuron.

<p align="center">
    <img src="https://raw.githubusercontent.com/JeffDeCola/my-cheat-sheets/refs/heads/master/docs/pics/software/development/neural-networks-mathematical-model-of-a-neuron.svg"
    alt="perceptron">
</p>

Let's create a structure to hold weights and bias,

type Perceptron struct {
    weights []float64
    bias    float64
}

Then we can create

## AND GATE EXAMPLE

For this example, we're going to model an AND gate (2 input and 1 output).

| INPUT 1 | INPUT 2 | OUTPUT |
|---------|---------|--------|
| 0       | 0       | 0      |
| 1       | 0       | 0      |
| 0       | 1       | 0      |
| 1       | 1       | 1      |
