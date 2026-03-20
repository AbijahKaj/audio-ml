/**
 * ASR demo placeholder: full browser run needs hosted SafeTensors + config + vocab (large assets).
 * See README and plans/fastconformer-asr.md for export and integration.
 */
export function createSpeechRecognizerDemo(container: HTMLElement): () => void {
  container.innerHTML = `
    <section class="app-explainer">
      <h2>Speech recognition (FastConformer)</h2>
      <p>
        The <code>SpeechRecognizer</code> application loads NeMo-exported SafeTensors and runs inference
        in the browser via TensorFlow.js (WASM). Host <code>model.safetensors</code>,
        <code>model_config.json</code>, and <code>vocab.json</code>, then call <code>load()</code> and
        <code>transcribe()</code> on 16&nbsp;kHz mono PCM.
      </p>
      <p class="app-explainer-note">
        This demo page does not bundle model weights. Use the Python tools in <code>tools/</code> to export
        assets and serve them from your site or CDN.
      </p>
    </section>
  `;
  return () => {
    container.innerHTML = '';
  };
}
