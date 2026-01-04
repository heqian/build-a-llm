import * as tf from "@tensorflow/tfjs";
import {
  SelfAttentionV1,
  SelfAttentionV2,
  CausalAttention,
  MultiHeadAttentionWrapper,
  MultiHeadAttention,
} from "./ch03";

const inputs = tf.tensor([
  [0.43, 0.15, 0.89], // Your     (x^1)
  [0.55, 0.87, 0.66], // journey  (x^2)
  [0.57, 0.85, 0.64], // starts   (x^3)
  [0.22, 0.58, 0.33], // with     (x^4)
  [0.77, 0.25, 0.1], // one      (x^5)
  [0.05, 0.8, 0.55], // step     (x^6)
]);

// Chapter 3.3
tf.tidy(() => {
  console.log("inputs:");
  inputs.print();
  console.log(inputs.shape);

  const query = inputs.gather([1]);
  console.log("query:");
  query.print();

  const attn_scores_2 = inputs.dot(query.transpose());
  console.log("attn_scores_2:");
  attn_scores_2.print();

  const attn_weights_2_tmp = attn_scores_2.div(attn_scores_2.sum());
  console.log("attn_weights_2_tmp:");
  attn_weights_2_tmp.print();

  const attn_weights_2 = attn_scores_2.softmax().expandDims(1);
  console.log("Attention weights:");
  attn_weights_2.print();
  console.log("Sum:");
  attn_weights_2.sum().print();

  const context_vec_2 = inputs.mul(attn_weights_2).sum(0, true);
  console.log("context_vec_2:");
  context_vec_2.print();

  // Chapter 3.3.2
  const attn_scores = inputs.matMul(inputs, false, true);
  console.log("attn_scores:");
  attn_scores.print();

  const attn_weights = attn_scores.softmax();
  console.log("attn_weights:");
  attn_weights.print();

  const all_context_vecs = attn_weights.matMul(inputs);
  console.log("all_context_vecs:");
  all_context_vecs.print();
  console.log("Previous 2nd context vector:", context_vec_2.dataSync());
});

// Chapter 3.4
tf.tidy(() => {
  const x_2 = inputs.gather([1]);
  console.log("second input element: x_2");
  x_2.print();
  const d_in = inputs.shape[1] as number;
  console.log(`the input embedding size, d=${d_in}`);
  const d_out = 2;
  console.log(`the output embedding size, d=${d_out}`);

  const seed = 123;
  // const initializer = tf.randomUniform([d_in, d_out], 0, 1, "float32", seed);
  // const W_query = tf.variable(initializer, false);
  // const W_key = tf.variable(initializer, false);
  // const W_value = tf.variable(initializer, false);
  const W_query = tf.tensor([
    [0.4541368, 0.0158234],
    [0.4371136, 0.2572921],
    [0.6334875, 0.4335591],
  ]);
  const W_key = tf.tensor([
    [0.4541368, 0.0158234],
    [0.4371136, 0.2572921],
    [0.6334875, 0.4335591],
  ]);
  const W_value = tf.tensor([
    [0.4541368, 0.0158234],
    [0.4371136, 0.2572921],
    [0.6334875, 0.4335591],
  ]);

  const query_2 = x_2.matMul(W_query);
  const key_2 = x_2.matMul(W_key);
  const value_2 = x_2.matMul(W_value);
  console.log("query_2:");
  query_2.print();

  const keys = inputs.matMul(W_key);
  const values = inputs.matMul(W_value);
  console.log("keys.shape:", keys.shape);
  console.log("values.shape:", values.shape);

  const keys_2 = keys.gather([1]);
  const attn_score_22 = query_2.dot(keys_2.transpose());
  attn_score_22.print();

  const attn_scores_2 = query_2.dot(keys.transpose());
  attn_scores_2.print();

  const d_k = keys.shape[1] as number;
  const attn_weights_2 = attn_scores_2.div(Math.sqrt(d_k)).softmax();
  console.log("attn_weights_2:");
  attn_weights_2.print();

  const context_vec_2 = attn_weights_2.matMul(values);
  console.log("context_vec_2:");
  context_vec_2.print();

  const sa_v1 = new SelfAttentionV1(d_in, d_out, seed);
  const output = sa_v1.apply(inputs) as tf.Tensor;
  console.log("SelfAttention_v1:");
  output.print();

  const sa_v2 = new SelfAttentionV2(d_in, d_out, seed);
  const output_v2 = sa_v2.apply(inputs) as tf.Tensor;
  console.log("SelfAttention_v2:");
  output_v2.print();
});

// Chapter 3.5
tf.tidy(() => {
  let d_in = inputs.shape[1] as number;
  let d_out = 2;
  const seed = 123;

  const sa_v2 = new SelfAttentionV2(d_in, d_out, seed);
  const queries = sa_v2.wQuery.apply(inputs) as tf.Tensor;
  const keys = sa_v2.wKey.apply(inputs) as tf.Tensor;
  const values = sa_v2.wValue.apply(inputs) as tf.Tensor;

  const attn_scores = queries.matMul(keys.transpose());
  let attn_weights = attn_scores
    .div(Math.sqrt(keys.shape[keys.shape.length - 1]))
    .softmax();
  console.log("attn_weights:");
  attn_weights.print();

  let context_length = attn_scores.shape[0];
  const mask_simple = tf.linalg.bandPart(
    tf.ones([context_length, context_length]),
    -1,
    0,
  );
  console.log("mask_simple:");
  mask_simple.print();

  const masked_simple = attn_weights.mul(mask_simple);
  console.log("masked_simple:");
  masked_simple.print();

  const row_sums = masked_simple.sum(-1, true);
  const masked_simple_norm = masked_simple.div(row_sums);
  console.log("masked_simple_norm:");
  masked_simple_norm.print();

  const mask = tf.linalg.bandPart(
    tf.ones([context_length, context_length]),
    -1,
    0,
  );
  const masked = tf.where(mask.equal(1), attn_scores, Number.NEGATIVE_INFINITY);
  console.log("masked:");
  masked.print();

  attn_weights = masked
    .div(Math.sqrt(keys.shape[keys.shape.length - 1]))
    .softmax();
  console.log("attn_weights:");
  attn_weights.print();

  console.log("attn_weights (with dropout rate: 0.5):");
  tf.dropout(attn_weights, 0.5).print();

  const batch = tf.stack([inputs, inputs]);
  console.log("batch.shape:", batch.shape);

  context_length = batch.shape[1] as number;
  const ca = new CausalAttention({
    dimIn: d_in,
    dimOut: d_out,
    contextLength: context_length,
    dropoutRate: 0.0,
    qkvBias: false,
  });
  let context_vecs = ca.apply(batch) as tf.Tensor;
  console.log("context_vecs:");
  context_vecs.print();
  console.log("context_vecs.shape:", context_vecs.shape);

  const mhaw = new MultiHeadAttentionWrapper({
    dimIn: d_in,
    dimOut: d_out,
    contextLength: context_length,
    dropoutRate: 0.0,
    numHeads: 2,
    qkvBias: false,
  });
  context_vecs = mhaw.apply(batch) as tf.Tensor;
  console.log("[mhaw] context_vecs (with numHeads = 2):");
  context_vecs.print();
  console.log(
    "[mhaw] context_vecs.shape (with numHeads = 2):",
    context_vecs.shape,
  );

  let batch_size = batch.shape[0] as number;
  context_length = batch.shape[1] as number;
  d_in = batch.shape[2] as number;
  d_out = 2;

  const mha = new MultiHeadAttention({
    dimIn: d_in,
    dimOut: d_out,
    contextLength: context_length,
    dropoutRate: 0.0,
    numHeads: 2,
    qkvBias: false,
  });
  context_vecs = mha.apply(batch) as tf.Tensor;
  console.log("[mha] context_vecs (with numHeads = 2):");
  context_vecs.print();
  console.log(
    "[mha] context_vecs.shape (with numHeads = 2):",
    context_vecs.shape,
  );
});
