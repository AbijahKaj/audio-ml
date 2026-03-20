import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { ConformerLayerWeights } from '../model/types';
import { ConvModule } from './ConvModule';
import { FeedForward } from './FeedForward';
import { RelPositionMultiHeadAttention } from './MultiHeadAttention';

export class ConformerBlock {
  private readonly ffn1: FeedForward;
  private readonly ffn2: FeedForward;
  private readonly attn: RelPositionMultiHeadAttention;
  private readonly conv: ConvModule;
  private readonly normSelfAtt: ConformerLayerWeights['normSelfAtt'];
  private readonly normOut: ConformerLayerWeights['normOut'];

  constructor(
    private readonly backend: ComputeBackend,
    w: ConformerLayerWeights,
    dModel: number,
    numHeads: number,
  ) {
    this.ffn1 = new FeedForward(
      backend,
      w.normFeedForward1,
      w.feedForward1.w1,
      w.feedForward1.w2,
    );
    this.ffn2 = new FeedForward(
      backend,
      w.normFeedForward2,
      w.feedForward2.w1,
      w.feedForward2.w2,
    );
    this.attn = new RelPositionMultiHeadAttention(backend, w.selfAtt, dModel, numHeads);
    this.conv = new ConvModule(backend, w.normConv, w.conv);
    this.normSelfAtt = w.normSelfAtt;
    this.normOut = w.normOut;
  }

  forward(x: TensorHandle, posEmb: TensorHandle): TensorHandle {
    let residual = x;

    let t = this.ffn1.forward(x);
    residual = this.backend.add(residual, this.backend.scale(t, 0.5));
    this.backend.dispose(t);

    let u = this.backend.layerNorm(
      residual,
      this.normSelfAtt.weight,
      this.normSelfAtt.bias,
      1e-5,
    );
    u = this.attn.forward(u, posEmb);
    residual = this.backend.add(residual, u);
    this.backend.dispose(u);

    const v = this.conv.forward(residual);
    residual = this.backend.add(residual, v);
    this.backend.dispose(v);

    let w = this.ffn2.forward(residual);
    residual = this.backend.add(residual, this.backend.scale(w, 0.5));
    this.backend.dispose(w);

    const out = this.backend.layerNorm(
      residual,
      this.normOut.weight,
      this.normOut.bias,
      1e-5,
    );
    this.backend.dispose(residual);
    return out;
  }
}