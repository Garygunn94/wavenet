from mdp import fastica
from scikits.audiolab import wavread, wavwrite
from numpy import abs, max
 
# Load in the stereo file
recording, fs, enc = wavread('./train/input/mixed_1.wav')
 
# Perform FastICA algorithm on the two channels
sources = fastica(recording)
 
# The output levels of this algorithm are arbitrary, so normalize them to 1.0.
sources /= max(abs(sources), axis = 0)
 
# Write back to a file
wavwrite(sources, 'sources.wav', fs, enc)