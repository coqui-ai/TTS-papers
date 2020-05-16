(Feel free to suggest changes)
## Papers
- Merging Phoneme and Char representations: https://arxiv.org/pdf/1811.07240.pdf
- Tacotron transfer learning : https://arxiv.org/pdf/1904.06508.pdf
- phoneme timing from attention: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8683827
- SEMI-SUPERVISED TRAINING FOR IMPROVING DATA EFFICIENCY IN END-TO-ENDSPEECH SYNTHESI - https://arxiv.org/pdf/1808.10128.pdf
- Listening while Speaking: Speech Chain by Deep Learning - https://arxiv.org/pdf/1707.04879.pdf
- FastSpeech: https://arxiv.org/abs/1905.09263
- GENERALIZED END-TO-END LOSS FOR SPEAKER VERIFICATION: https://arxiv.org/pdf/1710.10467.pdf
- Es-Tacotron2: Multi-Task Tacotron 2 with Pre-Trained Estimated Network for Reducing the Over-Smoothness Problem: https://www.mdpi.com/2078-2489/10/4/131/pdf
	- Against Over-Smoothness
- FastSpeech: https://arxiv.org/pdf/1905.09263.pdf
- Learning singing from speech: https://arxiv.org/pdf/1912.10128.pdf
- TTS-GAN: https://arxiv.org/pdf/1909.11646.pdf
    - they use duration and linguistic features for en2en TTS.
    - Close to WaveNet performance.
- DurIAN: https://arxiv.org/pdf/1909.01700.pdf
    - Duration aware Tacotron
- MelNet: https://arxiv.org/abs/1906.01083
- AlignTTS: https://arxiv.org/pdf/2003.01950.pdf
- Unsupervised Speech Decomposition via Triple Information Bottleneck
    - https://arxiv.org/pdf/2004.11284.pdf
    - https://anonymous0818.github.io/
- FlowTron: https://arxiv.org/pdf/2005.05957.pdf
    - Inverse Autoregresive Flow on Tacotron like architecture
    - WaveGlow as vocoder.
    - Speech style embedding with Mixture of Gaussian model.
    - Model is large and havier than vanilla Tacotron
    - MOS values are slighly better than public Tacotron implementation.
- Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention : https://arxiv.org/pdf/1710.08969.pdf 

## Multi-Speaker Papers
- Training Multi-Speaker Neural Text-to-Speech Systems using Speaker-Imbalanced Speech Corpora - https://arxiv.org/abs/1904.00771
- Deep Voice 2 - https://papers.nips.cc/paper/6889-deep-voice-2-multi-speaker-neural-text-to-speech.pdf
- Sample Efficient Adaptive TTS - https://openreview.net/pdf?id=rkzjUoAcFX
	- WaveNet + Speaker Embedding approach
- Voice Loop - https://arxiv.org/abs/1707.06588
- MODELING MULTI-SPEAKER LATENT SPACE TO IMPROVE NEURAL TTS QUICK ENROLLING NEW SPEAKER AND ENHANCING PREMIUM VOICE - https://arxiv.org/pdf/1812.05253.pdf
- Transfer learning from speaker verification to multispeaker text-to-speech synthesis - https://arxiv.org/pdf/1806.04558.pdf
- Fitting new speakers based on a short untranscribed sample - https://arxiv.org/pdf/1802.06984.pdf
- Generalized end-to-end loss for speaker verification

## Attention
- LOCATION-RELATIVE ATTENTION MECHANISMS FOR ROBUST LONG-FORMSPEECH SYNTHESIS : https://arxiv.org/pdf/1910.10288.pdf

## Vocoders
- MelGAN: https://arxiv.org/pdf/1910.06711.pdf
- ParallelWaveGAN: https://arxiv.org/pdf/1910.11480.pdf
    - Multi scale STFT loss
    - ~1M model parameters (very small)
    - Slightly worse than WaveRNN 
- Improving FFTNEt
    - https://www.okamotocamera.com/slt_2018.pdf
    - https://www.okamotocamera.com/slt_2018.pdf
- FFTnet
    - https://gfx.cs.princeton.edu/pubs/Jin_2018_FAR/clips/clips.php
    - https://gfx.cs.princeton.edu/pubs/Jin_2018_FAR/fftnet-jin2018.pdf
- SPEECH WAVEFORM RECONSTRUCTION USING CONVOLUTIONAL NEURALNETWORKS WITH NOISE AND PERIODIC INPUTS
    - 150.162.46.34:8080/icassp2019/ICASSP2019/pdfs/0007045.pdf
- Towards Achieveing Robust Universal Vocoding
    - https://arxiv.org/pdf/1811.06292.pdf
- LPCNet
    - https://arxiv.org/pdf/1810.11846.pdf
- ExciteNet
    - https://arxiv.org/pdf/1811.04769v3.pdf
- GELP: GAN-Excited Linear Prediction for Speech Synthesis fromMel-spectrogram
    - https://arxiv.org/pdf/1904.03976v3.pdf
- High Fidelity Speech Synthesis with Adversarial Networks: https://arxiv.org/abs/1909.11646
    - GAN-TTS, end-to-end speech synthesis
    - Uses duration and linguistic features
    - Duration and acoustic features are predicted by additional models.
    - Random Window Discriminator: Ingest not the whole Voice sample but random
    windows.
    - Multiple RWDs. Some conditional and some unconditional. (conditioned on 
    input features)
    - Punchline: Use randomly sampled windows with different window sizes for D.
    - Shared results sounds mechanical that shows the limits of non-neural
    acoustic features.
- Multi-Band MelGAN: https://arxiv.org/abs/2005.05106
    - Use PWGAN losses instead of feature-matching loss.
    - Using a larger receptive field boosts model performance significantly.
    - Generator pretraining for 200k iters.
    - Multi-Band voice signal prediction. The output is summation of 4 different
    band predictions with PQMF synthesis filters.
    - Multi-band model has 1.9m parameters (quite small).
    - Claimed to be 7x faster than MelGAN

## From the Internet (Blogs, Videos etc)

## Paper Discussion
- Tacotron 2 : https://www.youtube.com/watch?v=2iarxxm-v9w

## Talks
- End-to-End Text-to-Speech Synthesis, Part 1 : https://www.youtube.com/watch?v=RNKrq26Z0ZQ
- Speech synthesis from neural decoding of spoken sentences | AISC : https://www.youtube.com/watch?v=MNDtMDPmnMo
- Generative Text-to-Speech Synthesis : https://www.youtube.com/watch?v=j4mVEAnKiNg
- SPEECH SYNTHESIS FOR THE GAMING INDUSTRY : https://www.youtube.com/watch?v=aOHAYe4A-2Q

## General
- Modern Text-to-Speech Systems Review : https://www.youtube.com/watch?v=8rXLSc-ZcRY

