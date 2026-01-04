import * as tf from "@tensorflow/tfjs";

class SelfAttentionV1 extends tf.layers.Layer {
  private dIn: number;
  private dOut: number;
  private seed: number | undefined;

  // Weight parameters
  public wQuery!: tf.LayerVariable;
  public wKey!: tf.LayerVariable;
  public wValue!: tf.LayerVariable;

  constructor(dIn: number, dOut: number, seed?: number) {
    super();
    this.dIn = dIn;
    this.dOut = dOut;
    this.seed = seed;
  }

  build(inputShape: tf.Shape | tf.Shape[]): void {
    const initializer = tf.initializers.randomUniform({
      minval: 0,
      maxval: 1,
      seed: this.seed,
    });

    this.wQuery = this.addWeight(
      "w_query",
      [this.dIn, this.dOut],
      "float32",
      initializer,
    );

    this.wKey = this.addWeight(
      "w_key",
      [this.dIn, this.dOut],
      "float32",
      initializer,
    );

    this.wValue = this.addWeight(
      "w_value",
      [this.dIn, this.dOut],
      "float32",
      initializer,
    );
  }

  call(inputs: tf.Tensor | tf.Tensor[], kwargs?: any): tf.Tensor {
    return tf.tidy(() => {
      const input = Array.isArray(inputs) ? inputs[0] : inputs;

      // Matrix Multiplications: input @ W
      const keys = tf.matMul(input, this.wKey.read());
      const queries = tf.matMul(input, this.wQuery.read());
      const values = tf.matMul(input, this.wValue.read());

      // Attention Scores: queries @ keys.T
      // tf.matMul(a, b, false, true) transposes 'b' before multiplication
      const attnScores = tf.matMul(queries, keys, false, true);

      // Scale: scores / sqrt(d_out)
      // d_out corresponds to keys.shape[-1]
      const scale = Math.sqrt(this.dOut);
      const scaledScores = tf.div(attnScores, scale);

      // Softmax over the last dimension
      const attnWeights = tf.softmax(scaledScores, -1);

      // Context Vector: attn_weights @ values
      const contextVec = tf.matMul(attnWeights, values);

      return contextVec;
    });
  }

  // Required for TensorFlow.js to compute output shape automatically
  computeOutputShape(inputShape: tf.Shape): tf.Shape {
    // Input shape: [batch, sequence_len, d_in]
    // Output shape: [batch, sequence_len, d_out]
    const shape = [...(inputShape as number[])];
    shape[shape.length - 1] = this.dOut;
    return shape;
  }

  static get className() {
    return "SelfAttentionV1";
  }
}

class SelfAttentionV2 extends tf.layers.Layer {
  private dIn: number;
  private dOut: number;
  private qkvBias: boolean;

  // Sub-layers
  public wQuery: tf.layers.Layer;
  public wKey: tf.layers.Layer;
  public wValue: tf.layers.Layer;

  constructor(
    dIn: number,
    dOut: number,
    seed?: number,
    qkvBias: boolean = false,
  ) {
    super({});
    this.dIn = dIn;
    this.dOut = dOut;
    this.qkvBias = qkvBias;

    const kernelInitializer =
      typeof seed === "number"
        ? tf.initializers.randomUniform({
            seed: seed,
          })
        : undefined;

    // Initialize Linear (Dense) layers
    this.wQuery = tf.layers.dense({
      units: this.dOut,
      useBias: this.qkvBias,
      inputShape: [this.dIn], // Optional hint for the first build
      kernelInitializer,
    });

    this.wKey = tf.layers.dense({
      units: this.dOut,
      useBias: this.qkvBias,
      inputShape: [this.dIn],
      kernelInitializer,
    });

    this.wValue = tf.layers.dense({
      units: this.dOut,
      useBias: this.qkvBias,
      inputShape: [this.dIn],
      kernelInitializer,
    });
  }

  call(inputs: tf.Tensor | tf.Tensor[], kwargs?: any): tf.Tensor {
    return tf.tidy(() => {
      const input = Array.isArray(inputs) ? inputs[0] : inputs;

      // 1. Project inputs to Queries, Keys, and Values
      // Note: We cast to tf.Tensor to satisfy TypeScript type checking
      const keys = this.wKey.apply(input) as tf.Tensor;
      const queries = this.wQuery.apply(input) as tf.Tensor;
      const values = this.wValue.apply(input) as tf.Tensor;

      // 2. Compute Attention Scores
      // TFJS: tf.matMul(a, b, false, true) transposes 'b' before multiplication.
      const attnScores = tf.matMul(queries, keys, false, true);

      // 3. Scale and Softmax
      const keyDim = keys.shape[keys.shape.length - 1];
      const scale = Math.sqrt(keyDim);
      const scaledScores = attnScores.div(scale);
      const attnWeights = tf.softmax(scaledScores, -1);

      // 4. Compute Context Vector
      const contextVec = tf.matMul(attnWeights, values);

      return contextVec;
    });
  }

  // Required by TFJS to infer output shape during model compilation
  computeOutputShape(inputShape: tf.Shape): tf.Shape {
    // Input shape: [batch, sequence_len, d_in]
    // Output shape: [batch, sequence_len, d_out]
    const shape = [...(inputShape as number[])];
    shape[shape.length - 1] = this.dOut;
    return shape;
  }

  // Required for serialization (saving/loading models)
  static get className() {
    return "SelfAttentionV2";
  }
}

interface CausalAttentionConfig {
  dimIn: number;
  dimOut: number;
  contextLength: number;
  dropoutRate: number;
  qkvBias?: boolean;
}

class CausalAttention extends tf.layers.Layer {
  private wQuery: tf.layers.Layer;
  private wKey: tf.layers.Layer;
  private wValue: tf.layers.Layer;
  private dropoutLayer: tf.layers.Layer;
  private mask: tf.Tensor;

  constructor(config: CausalAttentionConfig) {
    super();

    // 1. Define Linear Projections (Dense layers)
    this.wQuery = tf.layers.dense({
      inputShape: [config.dimIn],
      units: config.dimOut,
      useBias: config.qkvBias,
    });
    this.wKey = tf.layers.dense({
      inputShape: [config.dimIn],
      units: config.dimOut,
      useBias: config.qkvBias,
    });
    this.wValue = tf.layers.dense({
      inputShape: [config.dimIn],
      units: config.dimOut,
      useBias: config.qkvBias,
    });

    // 2. Define Dropout
    this.dropoutLayer = tf.layers.dropout({ rate: config.dropoutRate });

    // 3. Create Causal Mask (Upper Triangular with diagonal=1)
    const ones = tf.ones([config.contextLength, config.contextLength]);
    const upper = tf.linalg.bandPart(ones, 0, -1); // Upper triangle including diag
    const diag = tf.linalg.bandPart(ones, 0, 0); // Just the diag

    // Result: 1s in the upper triangle (future), 0s elsewhere
    this.mask = tf.sub(upper, diag);
  }

  call(
    inputs: tf.Tensor | tf.Tensor[],
    kwargs: { training?: boolean } = {},
  ): tf.Tensor {
    return tf.tidy(() => {
      const input = Array.isArray(inputs) ? inputs[0] : inputs;

      // Shape: [batch, num_tokens, d_in]
      const [batch, numTokens, dIn] = input.shape;

      // Projections
      const keys = this.wKey.apply(input) as tf.Tensor;
      const queries = this.wQuery.apply(input) as tf.Tensor;
      const values = this.wValue.apply(input) as tf.Tensor;

      // Attention Scores
      let attnScores = tf.matMul(queries, keys, false, true);

      // Masking
      const currentMask = this.mask.slice([0, 0], [numTokens, numTokens]);
      // Note: We use -1e9 instead of -Infinity for numerical stability in JS softmax
      attnScores = tf.where(currentMask.toBool(), tf.scalar(-1e9), attnScores);

      // Scaling and Softmax
      const scale = tf.scalar(Math.sqrt(keys.shape[keys.shape.length - 1]));
      let attnWeights = tf.softmax(tf.div(attnScores, scale), -1);

      // Dropout
      // We must explicitly pass the training flag (kwargs['training'])
      attnWeights = this.dropoutLayer.apply(attnWeights, {
        training: kwargs.training,
      }) as tf.Tensor;

      // Context Vector: attn_weights @ values
      const contextVec = tf.matMul(attnWeights, values);

      return contextVec;
    });
  }

  // Necessary for serialization if you plan to save/load the model
  static get className() {
    return "CausalAttention";
  }
}

class MultiHeadAttentionWrapper extends tf.layers.Layer {
  private heads: tf.layers.Layer[];

  constructor(config: MultiHeadAttentionConfig) {
    super();

    this.heads = [];
    for (let i = 0; i < config.numHeads; i++) {
      this.heads.push(
        new CausalAttention({
          dimIn: config.dimIn,
          dimOut: config.dimOut,
          contextLength: config.contextLength,
          dropoutRate: config.dropoutRate,
          qkvBias: config.qkvBias,
        }),
      );
    }
  }

  call(
    inputs: tf.Tensor | tf.Tensor[],
    kwargs: { training?: boolean } = {},
  ): tf.Tensor {
    return tf.tidy(() => {
      const input = Array.isArray(inputs) ? inputs[0] : inputs;

      // 1. Run input through every head independently
      const headOutputs = this.heads.map(
        (head) => head.apply(input, { training: kwargs.training }) as tf.Tensor,
      );

      // 2. Concatenate results along the last dimension (dim = -1)
      return tf.concat(headOutputs, -1);
    });
  }

  static get className() {
    return "MultiHeadAttentionWrapper";
  }
}

interface MultiHeadAttentionConfig {
  dimIn: number;
  dimOut: number;
  contextLength: number;
  dropoutRate: number;
  numHeads: number;
  qkvBias?: boolean;
}

class MultiHeadAttention extends tf.layers.Layer {
  private dimOut: number;
  private numHeads: number;
  private headDim: number;
  private wQuery: tf.layers.Layer;
  private wKey: tf.layers.Layer;
  private wValue: tf.layers.Layer;
  private outputProjection: tf.layers.Layer;
  private dropoutLayer: tf.layers.Layer;
  private mask: tf.Tensor;

  constructor(config: MultiHeadAttentionConfig) {
    super();

    // Assert divisibility
    if (config.dimOut % config.numHeads !== 0) {
      throw new Error(
        `dOut (${config.dimOut}) must be divisible by numHeads (${config.numHeads})`,
      );
    }

    this.dimOut = config.dimOut;
    this.numHeads = config.numHeads;
    this.headDim = config.dimOut / config.numHeads;

    // Linear Layers
    this.wQuery = tf.layers.dense({
      inputDim: config.dimIn,
      units: config.dimOut,
      useBias: config.qkvBias,
    });
    this.wKey = tf.layers.dense({
      inputDim: config.dimIn,
      units: config.dimOut,
      useBias: config.qkvBias,
    });
    this.wValue = tf.layers.dense({
      inputDim: config.dimIn,
      units: config.dimOut,
      useBias: config.qkvBias,
    });
    this.outputProjection = tf.layers.dense({
      inputDim: config.dimOut,
      units: config.dimOut,
    }); // Combine heads

    this.dropoutLayer = tf.layers.dropout({ rate: config.dropoutRate });

    // Buffer: Mask
    // Create upper triangular matrix (future mask)
    const ones = tf.ones([config.contextLength, config.contextLength]);
    const upper = tf.linalg.bandPart(ones, 0, -1);
    const diag = tf.linalg.bandPart(ones, 0, 0);
    this.mask = tf.sub(upper, diag); // 1s in upper triangle, 0 elsewhere
  }

  call(
    inputs: tf.Tensor | tf.Tensor[],
    kwargs: { training?: boolean } = {},
  ): tf.Tensor {
    return tf.tidy(() => {
      const input = Array.isArray(inputs) ? inputs[0] : inputs;

      // Shape: [b, t, d_in]
      const [batch, numTokens, dimIn] = input.shape;

      // 1. Projections
      // Shape: [b, t, d_out]
      let keys = this.wKey.apply(input) as tf.Tensor;
      let queries = this.wQuery.apply(input) as tf.Tensor;
      let values = this.wValue.apply(input) as tf.Tensor;

      // 2. Split Heads & Transpose
      // We reshape from [b, t, d_out] -> [b, t, heads, d_head]
      // Then transpose to [b, heads, t, d_head] to align for matMul
      const splitHeads = (t: tf.Tensor) => {
        return t
          .reshape([batch, numTokens, this.numHeads, this.headDim])
          .transpose([0, 2, 1, 3]); // Swap dim 1 (tokens) and 2 (heads)
      };

      keys = splitHeads(keys);
      queries = splitHeads(queries);
      values = splitHeads(values);

      // 3. Scaled Dot-Product Attention
      // queries: [b, h, t, d]
      // keys:    [b, h, t, d]
      // Result:  [b, h, t, t]
      let attnScores = tf.matMul(queries, keys, false, true);

      // 4. Masking
      // Slice mask to current sequence length: [t, t]
      const currentMask = this.mask.slice([0, 0], [numTokens, numTokens]);

      // We need to reshape mask to broadcast correctly: [1, 1, t, t]
      // This ensures it applies to all batches and all heads
      const broadcastMask = currentMask.reshape([1, 1, numTokens, numTokens]);

      attnScores = tf.where(
        broadcastMask.toBool(),
        tf.scalar(-1e9),
        attnScores,
      );

      // 5. Softmax & Dropout
      const scale = tf.scalar(Math.sqrt(this.headDim));
      let attnWeights = tf.softmax(tf.div(attnScores, scale), -1);

      attnWeights = this.dropoutLayer.apply(attnWeights, {
        training: kwargs.training,
      }) as tf.Tensor;

      // 6. Context Vector
      // weights: [b, h, t, t]
      // values:  [b, h, t, d]
      // result:  [b, h, t, d]
      let contextVec = tf.matMul(attnWeights, values);

      // 7. Recombine Heads
      // Transpose back: [b, h, t, d] -> [b, t, h, d]
      contextVec = contextVec.transpose([0, 2, 1, 3]);

      // Reshape (flatten heads): [b, t, h * d] -> [b, t, d_out]
      contextVec = contextVec.reshape([batch, numTokens, this.dimOut]);

      // 8. Output Projection
      return this.outputProjection.apply(contextVec) as tf.Tensor;
    });
  }

  static get className() {
    return "MultiHeadAttention";
  }
}

tf.serialization.registerClass(CausalAttention);
tf.serialization.registerClass(MultiHeadAttention);

export {
  SelfAttentionV1,
  SelfAttentionV2,
  CausalAttention,
  MultiHeadAttentionWrapper,
  MultiHeadAttention,
};
