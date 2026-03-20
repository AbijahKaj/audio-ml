import { SpeechRecognizer } from '../dist/applications/index.js';

function decodePcm16Wav(buffer) {
  const view = new DataView(buffer);
  const channels = view.getUint16(22, true);
  const bitsPerSample = view.getUint16(34, true);
  let offset = 12;
  let dataOffset = 0;
  let dataSize = 0;

  while (offset < view.byteLength) {
    const chunkId = String.fromCharCode(
      view.getUint8(offset),
      view.getUint8(offset + 1),
      view.getUint8(offset + 2),
      view.getUint8(offset + 3),
    );
    const chunkSize = view.getUint32(offset + 4, true);

    if (chunkId === 'data') {
      dataOffset = offset + 8;
      dataSize = chunkSize;
      break;
    }

    offset += 8 + chunkSize + (chunkSize % 2);
  }

  if (!dataOffset || bitsPerSample !== 16) {
    throw new Error('Only PCM16 WAV files are supported by this smoke test');
  }

  const samples = new Int16Array(buffer, dataOffset, dataSize / 2);
  const frameCount = samples.length / channels;
  const mono = new Float32Array(frameCount);

  for (let index = 0; index < frameCount; index += 1) {
    let sum = 0;
    for (let channel = 0; channel < channels; channel += 1) {
      sum += samples[index * channels + channel] / 32768;
    }
    mono[index] = sum / channels;
  }

  return { audio: mono, sampleRate: view.getUint32(24, true) };
}

const url = process.argv[2] ?? 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav';
const response = await fetch(url);
if (!response.ok) {
  throw new Error(`Failed to fetch test audio: ${response.status} ${response.statusText}`);
}

const { audio, sampleRate } = decodePcm16Wav(await response.arrayBuffer());
const recognizer = new SpeechRecognizer({
  provider: 'transformers-js',
  modelId: 'Xenova/wav2vec2-base-960h',
  sampleRate,
  chunkLengthSeconds: 10,
  strideLengthSeconds: 2,
});

await recognizer.load();
const result = await recognizer.transcribe(audio);
console.log(JSON.stringify(result, null, 2));
