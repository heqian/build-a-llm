import * as _ from "lodash-es";
import * as tf from "@tensorflow/tfjs";
import { get_encoding } from "tiktoken";

class SimpleTokenizerV1 {
  private str_to_int: Record<string, number>;
  private int_to_str: Record<number, string>;

  constructor(vocab: Record<string, number>) {
    this.str_to_int = vocab;
    this.int_to_str = _.invert(vocab);
  }

  encode(text: string): number[] {
    const preprocessed = text
      .split(/([,.:;?_!"()\']|--|\s)/)
      .map((item) => item.trim())
      .filter((item) => item !== "");

    const ids = _.map(preprocessed, (str) => this.str_to_int[str]);
    return ids;
  }

  decode(ids: number[]): string {
    const text = _.chain(ids)
      .map((id) => this.int_to_str[id])
      .join(" ")
      .value();

    return text.replace(/\s+([,.?!"()'])/g, "$1");
  }
}

class SimpleTokenizerV2 {
  private str_to_int: Record<string, number>;
  private int_to_str: Record<number, string>;

  constructor(vocab: Record<string, number>) {
    this.str_to_int = vocab;
    this.int_to_str = _.invert(vocab);
  }

  encode(text: string): number[] {
    const preprocessed = text
      .split(/([,.:;?_!"()\']|--|\s)/)
      .map((item) => item.trim())
      .filter((item) => item !== "");

    const ids = _.map(preprocessed, (str) =>
      this.str_to_int[str] === undefined
        ? this.str_to_int["<|unk|>"]
        : this.str_to_int[str],
    );

    return ids;
  }

  decode(ids: number[]): string {
    const text = _.chain(ids)
      .map((id) => this.int_to_str[id])
      .join(" ")
      .value();

    return text.replace(/\s+([,.?!"()'])/g, "$1");
  }
}

function create_dataset_v1(
  txt: string,
  batch_size = 4,
  max_length = 256,
  stride = 128,
  shuffle = true,
) {
  const ids = [];

  const tokenizer = get_encoding("gpt2");
  const token_ids = tokenizer.encode(txt, ["<|unk|>"]);

  console.assert(
    token_ids.length > max_length,
    "Number of tokenized inputs must at least be equal to max_length+1",
  );

  for (let i = 0; i < token_ids.length - max_length; i += stride) {
    const input_chunk = token_ids.slice(i, i + max_length);
    const target_chunk = token_ids.slice(i + 1, i + max_length + 1);

    ids.push({
      input: new Int32Array(input_chunk.buffer),
      target: new Int32Array(target_chunk.buffer),
    });
  }

  const dataset = tf.data.array(ids).batch(batch_size);

  return shuffle ? dataset.shuffle(ids.length, undefined, true) : dataset;
}

export { SimpleTokenizerV1, SimpleTokenizerV2, create_dataset_v1 };
