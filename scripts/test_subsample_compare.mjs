import { readFileSync } from 'fs';
import { createTfjsBackend } from './init-tfjs-backend.mjs';
import { TfjsBackend } from '../packages/asr/src/compute/TfjsBackend';
import { loadSafeTensors } from '../packages/asr/src/model/SafeTensorsLoader';
import { parseModelConfig } from '../packages/asr/src/model/ModelConfig';
import { mapWeights } from '../packages/asr/src/model/WeightMapper';
import { ConvSubsampling } from '../packages/asr/src/encoder/ConvSubsampling';

async function main() {
  const backend = await createTfjsBackend(TfjsBackend);

  const config = parseModelConfig(readFileSync('/workspace/test_model/model_config.json', 'utf-8'));
  const rawWeights = await loadSafeTensors(readFileSync('/workspace/test_model/model.safetensors').buffer, backend);
  const modelWeights = mapWeights(rawWeights, config);
  const subsampling = new ConvSubsampling(backend, modelWeights.encoder.subsampling, config);

  const featBuf = readFileSync('/workspace/test_audio/libri_0_nemo_features.f32');
  const featData = new Float32Array(featBuf.buffer, featBuf.byteOffset, featBuf.byteLength / 4);
  const features = backend.tensor(featData, [1, 586, 80]);

  const subOut = subsampling.forward(features);
  const jsData = await backend.getData(subOut);
  const jsShape = backend.getShape(subOut);
  console.log('JS subsampling: [' + jsShape.join(', ') + ']');

  const nemoBuf = readFileSync('/workspace/test_audio/nemo_pre_encode.f32');
  const nemoData = new Float32Array(nemoBuf.buffer, nemoBuf.byteOffset, nemoBuf.byteLength / 4);

  let jsMin=Infinity, jsMax=-Infinity;
  for (let i = 0; i < jsData.length; i++) {
    jsMin = Math.min(jsMin, jsData[i]); jsMax = Math.max(jsMax, jsData[i]);
  }
  let nMin=Infinity, nMax=-Infinity;
  for (let i = 0; i < nemoData.length; i++) {
    nMin = Math.min(nMin, nemoData[i]); nMax = Math.max(nMax, nemoData[i]);
  }
  console.log('  JS: min=' + jsMin.toFixed(1) + ', max=' + jsMax.toFixed(1));
  console.log('  NeMo: min=' + nMin.toFixed(1) + ', max=' + nMax.toFixed(1));

  console.log('First 5 of frame 0:');
  console.log('  JS:  ', Array.from(jsData.slice(0,5)).map(v=>v.toFixed(2)));
  console.log('  NeMo:', Array.from(nemoData.slice(0,5)).map(v=>v.toFixed(2)));

  let dot=0, na=0, nb=0;
  for (let d=0; d<512; d++) {
    dot += jsData[d]*nemoData[d]; na += jsData[d]**2; nb += nemoData[d]**2;
  }
  console.log('Cosine sim (frame 0):', (dot/(Math.sqrt(na)*Math.sqrt(nb))).toFixed(6));
}

main().catch(e => { console.error(e.message); process.exit(1); });
