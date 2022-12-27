# MSS_221118_project

#Model name: Parallel stacked hourglass network

A multiband spectrogram is used as input to PSHN. Spectrograms represent similar patterns in different frequency bands. Especially, low frequencies are bounded by high energies, while, higher frequencies contain low energies and noise. In consequence, the spectrogram is divided into equal halves along the frequency axis to form two sub-band spectrograms and a different convolution filter is applied in each band.


The parts of the stacked hourglass network (SHN) that receive the input from the upper band, lower band, and full band, and those are passing through an upper band stacked hourglass network (UBSHN), lower band stacked hourglass network (LBSHN), and full band stacked hourglass network (FBSHN), respectively. The combination of all three is a PSHN. Each of the UBSHN, LBSHN, and FBSHN contains four stacked hourglass modules in total.

The process of estimating the masks and predicting the spectrograms between an earlier and a subsequent parallel hourglass module is referred to as intermediate predictions. There are four stacks of hourglass modules for the upper band, lower band, and full band spectrograms. Accordingly, we needed to calculate four losses: three for the intermediate predictions and one for the final prediction.

The overall architecture of the model is shown in following figure

![image](https://user-images.githubusercontent.com/26374302/209610669-d84869af-2b5b-456c-a9b3-72ecbd0b9ab6.png)
