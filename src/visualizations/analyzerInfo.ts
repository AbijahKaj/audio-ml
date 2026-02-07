/**
 * Information about each analyzer for tooltips/info displays
 */

export interface AnalyzerInfo {
  name: string;
  description: string;
  purpose: string;
  computation: string;
  resources: Array<{ title: string; url: string }>;
}

export const analyzerInfoMap: Map<string, AnalyzerInfo> = new Map([
  ['MFCC', {
    name: 'Mel-Frequency Cepstral Coefficients',
    description: 'MFCCs are features widely used in speech and audio recognition that represent the short-term power spectrum of sound on a mel scale.',
    purpose: 'Used for speech recognition, speaker identification, music information retrieval, and audio classification tasks.',
    computation: '1. Compute FFT of audio frame 2. Apply mel filter bank 3. Take logarithm 4. Apply Discrete Cosine Transform (DCT)',
    resources: [
      { title: 'Wikipedia - MFCC', url: 'https://en.wikipedia.org/wiki/Mel-frequency_cepstrum' },
      { title: 'Speech Recognition Tutorial', url: 'https://www.speech.cs.cmu.edu/15-492/slides/03_mfcc.pdf' }
    ]
  }],
  ['PLP', {
    name: 'Perceptual Linear Prediction',
    description: 'PLP is a speech analysis technique that models the auditory spectrum using perceptual principles, combining critical band analysis with linear prediction.',
    purpose: 'Used in speech recognition and speaker verification systems, particularly effective for noisy environments.',
    computation: '1. Critical band analysis (Bark scale) 2. Equal loudness pre-emphasis 3. Intensity-loudness conversion 4. Autocorrelation 5. LPC analysis',
    resources: [
      { title: 'Wikipedia - PLP', url: 'https://en.wikipedia.org/wiki/Perceptual_linear_prediction' },
      { title: 'Original PLP Paper', url: 'https://ieeexplore.ieee.org/document/18663' }
    ]
  }],
  ['Chroma Features', {
    name: 'Chroma Features',
    description: 'Chroma features represent the tonal content of audio by mapping all frequencies to 12 pitch classes (C, C#, D, ..., B), ignoring octave information.',
    purpose: 'Used for music information retrieval, chord recognition, key detection, and audio similarity matching.',
    computation: '1. Compute FFT 2. Map frequencies to MIDI notes 3. Aggregate energy into 12 chroma bins (one per pitch class) 4. Normalize',
    resources: [
      { title: 'Wikipedia - Chroma Feature', url: 'https://en.wikipedia.org/wiki/Chroma_feature' },
      { title: 'Music Information Retrieval', url: 'https://www.audiolabs-erlangen.de/resources/MIR/FMP/C3/C3.html' }
    ]
  }],
  ['LPC', {
    name: 'Linear Predictive Coding',
    description: 'LPC models a signal as a linear combination of its previous samples, extracting coefficients that represent the spectral envelope.',
    purpose: 'Used in speech coding, synthesis, recognition, and audio compression.',
    computation: '1. Compute autocorrelation 2. Apply Levinson-Durbin recursion to solve for LPC coefficients',
    resources: [
      { title: 'Wikipedia - LPC', url: 'https://en.wikipedia.org/wiki/Linear_predictive_coding' },
      { title: 'LPC Tutorial', url: 'https://ccrma.stanford.edu/~hskim08/lpc/' }
    ]
  }],
  ['Constant-Q Transform', {
    name: 'Constant-Q Transform',
    description: 'CQT is a time-frequency analysis method with logarithmically spaced frequency bins, providing constant frequency resolution per octave.',
    purpose: 'Used in music analysis, pitch detection, and audio feature extraction where logarithmic frequency spacing is beneficial.',
    computation: '1. Create logarithmically spaced frequency bins 2. Apply Gaussian-like kernels in frequency domain 3. Compute magnitude spectrum',
    resources: [
      { title: 'Wikipedia - Constant-Q Transform', url: 'https://en.wikipedia.org/wiki/Constant-Q_transform' },
      { title: 'CQT Paper', url: 'https://ieeexplore.ieee.org/document/481513' }
    ]
  }],
  ['Wavelet Transform', {
    name: 'Wavelet Transform',
    description: 'Wavelet transform decomposes a signal into different frequency components using wavelets, providing both time and frequency information.',
    purpose: 'Used for signal denoising, compression, feature extraction, and multi-resolution analysis.',
    computation: '1. Apply Haar wavelet filters 2. Decompose into approximation and detail coefficients 3. Recursively decompose approximation (multi-level)',
    resources: [
      { title: 'Wikipedia - Wavelet Transform', url: 'https://en.wikipedia.org/wiki/Wavelet_transform' },
      { title: 'Wavelet Tutorial', url: 'https://www.mathworks.com/help/wavelet/getting-started-with-wavelet-toolbox.html' }
    ]
  }],
  ['Waveform Envelope', {
    name: 'Waveform Envelope',
    description: 'The amplitude envelope tracks the maximum absolute amplitude in a sliding window, showing the overall shape of the signal amplitude over time.',
    purpose: 'Used for onset detection, dynamics analysis, and visualizing signal dynamics.',
    computation: '1. Apply sliding window 2. Compute maximum absolute amplitude in each window 3. Output envelope values',
    resources: [
      { title: 'Envelope Detection', url: 'https://en.wikipedia.org/wiki/Envelope_detector' }
    ]
  }],
  ['Autocorrelation', {
    name: 'Autocorrelation',
    description: 'Autocorrelation measures the similarity of a signal with a delayed copy of itself, revealing periodic patterns and fundamental frequency.',
    purpose: 'Used for pitch detection, periodicity analysis, tempo estimation, and fundamental frequency extraction.',
    computation: '1. For each lag, compute correlation between signal and shifted version 2. Normalize by number of overlapping samples',
    resources: [
      { title: 'Wikipedia - Autocorrelation', url: 'https://en.wikipedia.org/wiki/Autocorrelation' },
      { title: 'Pitch Detection', url: 'https://cnx.org/contents/8QNnmdpE@2/Pitch-Detection-Algorithms' }
    ]
  }],
  ['Spectral Centroid', {
    name: 'Spectral Centroid',
    description: 'The spectral centroid is the "center of mass" of the spectrum, representing the frequency around which the spectral energy is concentrated.',
    purpose: 'Used as a measure of brightness or timbre, useful for music classification and audio analysis.',
    computation: '1. Compute magnitude spectrum 2. Calculate weighted average of frequencies (weighted by magnitude)',
    resources: [
      { title: 'Spectral Centroid', url: 'https://en.wikipedia.org/wiki/Spectral_centroid' },
      { title: 'Audio Features', url: 'https://musicinformationretrieval.com/audio_features.html' }
    ]
  }],
  ['Spectral Rolloff', {
    name: 'Spectral Rolloff',
    description: 'Spectral rolloff is the frequency below which a specified percentage (typically 85%) of the total spectral energy is contained.',
    purpose: 'Used to distinguish between noise-like and tone-like sounds, useful for audio classification.',
    computation: '1. Compute magnitude spectrum 2. Calculate cumulative energy 3. Find frequency where cumulative energy reaches threshold (e.g., 85%)',
    resources: [
      { title: 'Spectral Rolloff', url: 'https://musicinformationretrieval.com/spectral_features.html' }
    ]
  }],
  ['Spectral Bandwidth', {
    name: 'Spectral Bandwidth',
    description: 'Spectral bandwidth measures the spread of the spectrum around the centroid, indicating how concentrated or spread out the spectral energy is.',
    purpose: 'Used to characterize timbre and distinguish between different sound types.',
    computation: '1. Compute spectral centroid 2. Calculate weighted standard deviation of frequencies from centroid',
    resources: [
      { title: 'Spectral Bandwidth', url: 'https://musicinformationretrieval.com/spectral_features.html' }
    ]
  }],
  ['Spectral Flatness', {
    name: 'Spectral Flatness',
    description: 'Spectral flatness measures how noise-like a sound is by comparing the geometric mean to the arithmetic mean of the spectrum.',
    purpose: 'Used to distinguish between tonal and noise-like sounds, useful for audio classification.',
    computation: '1. Compute geometric mean of magnitude spectrum 2. Compute arithmetic mean 3. Ratio of geometric to arithmetic mean',
    resources: [
      { title: 'Spectral Flatness', url: 'https://en.wikipedia.org/wiki/Spectral_flatness' }
    ]
  }],
  ['Zero Crossing Rate', {
    name: 'Zero Crossing Rate',
    description: 'ZCR measures the rate at which the signal changes sign (crosses zero), indicating the frequency content and voicing characteristics.',
    purpose: 'Used to distinguish between voiced and unvoiced speech, detect percussive sounds, and classify audio.',
    computation: 'Count the number of sign changes (zero crossings) per unit time',
    resources: [
      { title: 'Zero Crossing Rate', url: 'https://en.wikipedia.org/wiki/Zero-crossing_rate' },
      { title: 'Audio Features Tutorial', url: 'https://musicinformationretrieval.com/audio_features.html' }
    ]
  }],
  ['RMSE', {
    name: 'Root Mean Square Energy',
    description: 'RMSE (or RMS) is a measure of the average power or energy of a signal, representing the overall loudness or amplitude level.',
    purpose: 'Used for silence detection, loudness measurement, dynamic range analysis, and audio level monitoring.',
    computation: '1. Square each sample 2. Compute mean 3. Take square root',
    resources: [
      { title: 'RMS Energy', url: 'https://en.wikipedia.org/wiki/Root_mean_square' },
      { title: 'Audio Level Measurement', url: 'https://en.wikipedia.org/wiki/Audio_level' }
    ]
  }],
  ['Mel Spectrogram', {
    name: 'Mel Spectrogram',
    description: 'Mel spectrogram represents the power spectrum of audio on a mel scale, which approximates human auditory perception of frequency.',
    purpose: 'Used in speech recognition, music information retrieval, and as input features for deep learning models.',
    computation: '1. Compute FFT 2. Apply mel filter bank 3. Take logarithm of mel energies',
    resources: [
      { title: 'Mel Scale', url: 'https://en.wikipedia.org/wiki/Mel_scale' },
      { title: 'Spectrogram', url: 'https://en.wikipedia.org/wiki/Spectrogram' }
    ]
  }],
  ['FFT', {
    name: 'Fast Fourier Transform',
    description: 'FFT computes the frequency domain representation of a time-domain signal, showing the magnitude of each frequency component.',
    purpose: 'Used for frequency analysis, spectrum visualization, and as a foundation for many other audio analysis techniques.',
    computation: '1. Apply window function (optional) 2. Compute FFT 3. Calculate magnitude spectrum from complex values',
    resources: [
      { title: 'Wikipedia - FFT', url: 'https://en.wikipedia.org/wiki/Fast_Fourier_transform' },
      { title: 'FFT Tutorial', url: 'https://betterexplained.com/articles/an-interactive-guide-to-the-fourier-transform/' }
    ]
  }]
]);
