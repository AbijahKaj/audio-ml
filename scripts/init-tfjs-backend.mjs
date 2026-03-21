/**
 * For Node benchmark/test scripts: use native TensorFlow when @tensorflow/tfjs-node
 * is installed (optional dependency of @audio-ml/asr), otherwise pure JS CPU.
 */
export async function createTfjsBackend(TfjsBackend) {
  const backend = new TfjsBackend();
  try {
    await backend.init('tensorflow');
  } catch {
    await backend.init('cpu');
  }
  return backend;
}
