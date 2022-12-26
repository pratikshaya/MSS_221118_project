# MSS_221118_project

Model name
Parallel stacked hourglass network

A multiband spectrogram is used as input to PSHN. Spectrograms represent similar patterns in different frequency bands. Especially, low frequencies are bounded by high energies, while, higher frequencies contain low energies and noise. In consequence, the spectrogram is divided into equal halves along the frequency axis to form two sub-band spectrograms and a different convolution filter is applied in each band.
The parts of the stacked hourglass network (SHN) that receive the input from the upper band, lower band, and full band, and those are passing through an upper band stacked hourglass network (UBSHN), lower band stacked hourglass network (LBSHN), and full band stacked hourglass network (FBSHN), respectively. The combination of all three is a PSHN. Each of the UBSHN, LBSHN, and FBSHN contains four stacked hourglass modules in total.
