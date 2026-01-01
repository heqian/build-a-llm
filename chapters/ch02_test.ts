import * as _ from "lodash-es";
import * as tf from "@tensorflow/tfjs";
import { get_encoding } from "tiktoken";

import {
  SimpleTokenizerV1,
  SimpleTokenizerV2,
  create_dataset_v1,
} from "./ch02";

// Chapter 2.2
async function loadText() {
  const response = await fetch(
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt",
  );
  return response.text();
}

const story = await loadText();

console.log("Total number of characters:", story.length);
console.log(story.substring(0, 99));

const preprocessed = story
  .split(/([,.:;?_!"()\']|--|\s)/)
  .map((word) => word.trim())
  .filter((item) => item !== "");
console.log(preprocessed.slice(0, 30));
console.log(preprocessed.length);

// Chapter 2.3
const all_words = _.chain(preprocessed).uniq().sort().value();
console.log(all_words.length);

let vocab = _.fromPairs(all_words.map((word, index) => [word, index]));
for (const word of Object.keys(vocab)) {
  console.log(word, vocab[word]);
  if (vocab[word] >= 50) break;
}

const tokenizer_v1 = new SimpleTokenizerV1(vocab);
const ids = tokenizer_v1.encode(
  `"It's the last he painted, you know,"
             Mrs. Gisburn said with pardonable pride.`,
);
console.log(ids);
console.log(tokenizer_v1.decode(ids));

// Chapter 2.4
let all_tokens = _.chain(preprocessed).uniq().sort().value();
all_tokens = all_tokens.concat(["<|endoftext|>", "<|unk|>"]);
console.log(all_tokens.length);

vocab = _.fromPairs(all_tokens.map((token, index) => [token, index]));
console.log(Object.entries(vocab).slice(-5));

const text1 = "Hello, do you like tea?";
const text2 = "In the sunlit terraces of the palace.";
let text = [text1, text2].join(" <|endoftext|> ");
console.log(text);

const tokenizer_v2 = new SimpleTokenizerV2(vocab);
console.log(tokenizer_v2.encode(text));
console.log(tokenizer_v2.decode(tokenizer_v2.encode(text)));

// Chapter 2.5
const tokenizer = get_encoding("gpt2");
text =
  "Hello, do you like tea? <|endoftext|> In the sunlit terracesof someunknownPlace.";
const integers = tokenizer.encode(text, ["<|endoftext|>"]);
console.log(integers);
console.log(new TextDecoder().decode(tokenizer.decode(integers)));

// Chapter 2.6
const enc_text = tokenizer.encode(story);
console.log(enc_text.length);

const enc_sample = enc_text.slice(50);
const context_size = 4;
const x = enc_sample.slice(0, context_size);
const y = enc_sample.slice(1, context_size + 1);
console.log(`x: ${x}`);
console.log(`y:     ${y}`);

for (let i = 1; i < context_size + 1; i++) {
  const context = enc_sample.slice(0, i);
  const desired = enc_sample.slice(i, i + 1);

  console.log(`${context} ----> ${desired}`);
}

const textDecoder = new TextDecoder();
for (let i = 1; i < context_size + 1; i++) {
  const context = enc_sample.slice(0, i);
  const desired = enc_sample.slice(i, i + 1);

  console.log(
    `${textDecoder.decode(tokenizer.decode(context))} ----> ${textDecoder.decode(tokenizer.decode(desired))}`,
  );
}

let dataset = create_dataset_v1(story, 1, 4, 1, false);
let data_iter = await dataset.iterator();
const first_batch = await data_iter.next();
console.log(
  `input: ${await first_batch.value.input.data()}, target: ${await first_batch.value.target.data()}`,
);
const second_batch = await data_iter.next();
console.log(
  `input: ${await second_batch.value.input.data()}, target: ${await second_batch.value.target.data()}`,
);

dataset = create_dataset_v1(story, 8, 4, 4, false);
data_iter = await dataset.iterator();
let batch = await data_iter.next();
const { input, target } = batch.value;
console.log("Input:");
input.print();
target.print();

// Chapter 2.7
const input_ids = tf.tensor([2, 3, 5, 1]);
let vocab_size = 6;
let output_dim = 3;

const embedding = tf.layers.embedding({
  inputDim: vocab_size,
  outputDim: output_dim,
});

const output = embedding.apply(input_ids);
console.log("Embedding layer weights:");
embedding.getWeights()[0].print();
console.log("Embeddings for four ids:");
output.print();

// Chapter 2.8
vocab_size = 50257;
output_dim = 256;
const token_embedding_layer = tf.layers.embedding({
  inputDim: vocab_size,
  outputDim: output_dim,
});

const max_length = 4;
dataset = create_dataset_v1(story, 8, max_length, max_length, false);
data_iter = await dataset.iterator();
batch = await data_iter.next();

const inputs = batch.value.input;
const targets = batch.value.target;
console.log("Token IDs:");
inputs.print();
console.log("Inputs shape:", inputs.shape);

const token_embeddings = token_embedding_layer.apply(inputs);
console.log("Token embeddings shape:", token_embeddings.shape);

const context_length = max_length;
const pos_embedding_layer = tf.layers.embedding({
  inputDim: context_length,
  outputDim: output_dim,
});
const pos_embeddings = pos_embedding_layer.apply(tf.range(0, context_length));
console.log("Position embeddings shape:", pos_embeddings.shape);

const input_embeddings = tf.add(token_embeddings, pos_embeddings);
console.log("Input embeddings shape:", input_embeddings.shape);

export {};
