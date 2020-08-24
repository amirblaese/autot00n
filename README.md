# Autot00n

## About

Simple autotune script to tune .wav files. Works best with monophonous sources.

## Dependencies

Scipy(rfft,resample), Soundfile, Numpy

## How to use?

Simply enter the name of your file, then enter the desired scale and the block length that is tuned. If source is not
mono, an option will be provided to sum to mono.

## How does it work?

The data is first chopped into finite sized blocks. Then an FFT is used to find the peak using a basic peak find (find amplitude). If the peak is not in agreement with the nearest scale frequency, it will be resampled such that the peak is that nearest scale frequency. The pieces are then resynthesized to a tuned .wav file.