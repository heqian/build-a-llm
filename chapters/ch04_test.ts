import * as tf from "@tensorflow/tfjs";
import { GPTConfig } from "./ch04";
import { DummyGPTModel } from "./ch04";

// Example usage:
const config: GPTConfig = {
  vocabSize: 50257,
  embeddingDim: 768,
  contextLength: 1024,
  dropRate: 0.1,
  numLayers: 12,
};

const model = new DummyGPTModel(config);

// Example forward pass
const batchSize = 2;
const seqLen = 10;
const inputIds = tf.randomUniform(
  [batchSize, seqLen],
  0,
  config.vocabSize,
  "int32",
);
const logits = model.apply(inputIds, { training: true }) as tf.Tensor;

console.log("Input shape:", inputIds.shape);
console.log("Output shape:", logits.shape);

// Clean up
inputIds.dispose();
logits.dispose();
