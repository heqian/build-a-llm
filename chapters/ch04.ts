import * as tf from "@tensorflow/tfjs";

interface GPTConfig {
  vocabSize: number;
  embeddingDim: number;
  contextLength: number;
  dropRate: number;
  numLayers: number;
}

class DummyGPTModel extends tf.layers.Layer {
  private tokenEmbedding: tf.layers.Layer;
  private positionEmbedding: tf.layers.Layer;
  private dropoutLayer: tf.layers.Layer;
  private transformerBlocks: DummyTransformerBlock[];
  private finalNorm: DummyLayerNorm;
  private outHead: tf.layers.Layer;

  constructor(config: GPTConfig) {
    super();

    // Token Embeddings
    this.tokenEmbedding = tf.layers.embedding({
      inputDim: config.vocabSize,
      outputDim: config.embeddingDim,
      name: "token_embedding",
    });

    // Positional Embeddings
    this.positionEmbedding = tf.layers.embedding({
      inputDim: config.contextLength,
      outputDim: config.embeddingDim,
      name: "position_embedding",
    });

    // Dropout
    this.dropoutLayer = tf.layers.dropout({
      rate: config.dropRate,
      name: "dropout",
    });

    // Transformer Blocks
    this.transformerBlocks = [];
    for (let i = 0; i < config.numLayers; i++) {
      this.transformerBlocks.push(new DummyTransformerBlock(config));
    }

    // Final Layer Norm
    this.finalNorm = new DummyLayerNorm(config.embeddingDim);

    // Output Head (Linear)
    this.outHead = tf.layers.dense({
      inputDim: config.embeddingDim,
      units: config.vocabSize,
      useBias: false,
      name: "output_head",
    });
  }

  call(
    inputs: tf.Tensor | tf.Tensor[],
    kwargs: { training?: boolean } = {},
  ): tf.Tensor {
    return tf.tidy(() => {
      const input = Array.isArray(inputs) ? inputs[0] : inputs;
      const training = kwargs?.training ?? false;
      const [batch, numTokens] = input.shape;

      // Token embeddings
      const tokenEmbeddings = this.tokenEmbedding.apply(input) as tf.Tensor;

      // Position embeddings
      const positionIndices = tf.range(0, numTokens, 1, "int32");
      const positionEmbeddings = this.positionEmbedding.apply(
        positionIndices,
      ) as tf.Tensor;

      // Add token and position embeddings
      let x = tf.add(tokenEmbeddings, positionEmbeddings);

      // Apply dropout
      x = this.dropoutLayer.apply(x, { training }) as tf.Tensor;

      // Apply transformer blocks
      for (const block of this.transformerBlocks) {
        x = block.apply(x, { training }) as tf.Tensor;
      }

      // Apply final layer norm
      x = this.finalNorm.apply(x) as tf.Tensor;

      // Apply output head
      const logits = this.outHead.apply(x) as tf.Tensor;

      return logits;
    });
  }

  static get className() {
    return "DummyGPTModel";
  }
}

class DummyTransformerBlock extends tf.layers.Layer {
  private config: GPTConfig;

  constructor(config: GPTConfig) {
    super();
    this.config = config;
  }

  call(
    inputs: tf.Tensor | tf.Tensor[],
    kwargs: { training?: boolean } = {},
  ): tf.Tensor {
    const input = Array.isArray(inputs) ? inputs[0] : inputs;
    return input;
  }

  static get className() {
    return "DummyTransformerBlock";
  }
}

class DummyLayerNorm extends tf.layers.Layer {
  private normalizedShape: number;
  private eps: number;

  constructor(normalizedShape: number, eps: number = 1e-5) {
    super();
    this.normalizedShape = normalizedShape;
    this.eps = eps;
  }

  call(
    inputs: tf.Tensor | tf.Tensor[],
    kwargs: { training?: boolean } = {},
  ): tf.Tensor {
    const input = Array.isArray(inputs) ? inputs[0] : inputs;
    return input;
  }

  static get className() {
    return "DummyLayerNorm";
  }
}

export { GPTConfig, DummyGPTModel, DummyTransformerBlock, DummyLayerNorm };
