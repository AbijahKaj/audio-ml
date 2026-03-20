import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { FastConformerConfig } from '../model/ModelConfig';

export abstract class TransducerDecoder {
  protected readonly blankId = 0;
  protected readonly maxSymbolsPerStep = 8;

  constructor(
    protected readonly backend: ComputeBackend,
    protected readonly config: FastConformerConfig,
  ) {}

  abstract decode(encoderOutput: TensorHandle): Promise<number[]>;

  protected async argMax(tensor: TensorHandle): Promise<number> {
    const data = await this.backend.getData(tensor);
    let bestIndex = 0;
    let bestValue = Number.NEGATIVE_INFINITY;
    for (let index = 0; index < data.length; index += 1) {
      if (data[index] > bestValue) {
        bestValue = data[index];
        bestIndex = index;
      }
    }
    return bestIndex;
  }
}
