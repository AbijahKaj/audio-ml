import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { JointNetworkWeights } from '../model/WeightMapper';
import { JointNetwork } from './JointNetwork';
import { Linear } from '../encoder/Linear';

/**
 * TDT joint network. The output projection produces [vocab_size + num_durations]
 * logits in a single linear layer. We split them into token and duration logits.
 */
export class TDTJointNetwork extends JointNetwork {
  private outputProj: Linear;
  private vocabSize: number;
  private numDurations: number;

  constructor(
    backend: ComputeBackend,
    weights: JointNetworkWeights,
    vocabSize: number,
    numDurations: number,
  ) {
    super(backend, weights);
    this.outputProj = new Linear(backend, weights.outputProj);
    this.vocabSize = vocabSize;
    this.numDurations = numDurations;
  }

  forward(
    encoderFrame: TensorHandle,
    predictionOut: TensorHandle
  ): { tokenLogits: TensorHandle; durationLogits: TensorHandle } {
    const joint = this.computeJoint(encoderFrame, predictionOut);
    const combinedLogits = this.outputProj.forward(joint);
    this.backend.dispose(joint);

    // Split into token logits and duration logits
    const [tokenLogits, durationLogits] = this.backend.split(
      combinedLogits, [this.vocabSize, this.numDurations], -1
    );
    this.backend.dispose(combinedLogits);

    return { tokenLogits, durationLogits };
  }
}
