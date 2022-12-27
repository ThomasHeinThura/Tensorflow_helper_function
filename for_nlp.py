# for nlp
import tensorflow as tf

def make_simple_nlp_model():
    # Set random seed and create embedding layer (new embedding layer for each model)
    tf.random.set_seed(42)
    max_vocab_length = 10000 # max number of words to have in our vocabulary
    max_length = 15 # max length our sequences will be (e.g. how many words from a Tweet does our model see?)

    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
    model_embedding = layers.Embedding(input_dim=max_vocab_length,
                                        output_dim=128,
                                        embeddings_initializer="uniform",
                                        input_length=max_length,
                                        name="embedding_2")
    text_vectorizer = TextVectorization(max_tokens=max_vocab_length,
                                    output_mode="int",
                                    output_sequence_length=max_length)

    # Create LSTM model
    inputs = layers.Input(shape=(1,), dtype="string")
    x = text_vectorizer(inputs)
    x = model_embedding(x)
    print(x.shape)
    # x = layers.LSTM(64, return_sequences=True)(x) # return vector for each word in the Tweet (you can stack RNN cells as long as return_sequences=True)
    x = layers.LSTM(64)(x) # return vector for whole sequence
    print(x.shape)
    # x = layers.Dense(64, activation="relu")(x) # optional dense layer on top of output of LSTM cell
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs, name="model_2_LSTM")
    return model

# Create a helper function to compare our baseline results to new model results
def compare_baseline_to_new_results(baseline_results, new_model_results):
  for key, value in baseline_results.items():
    print(f"Baseline {key}: {value:.2f}, New {key}: {new_model_results[key]:.2f}, Difference: {new_model_results[key]-value:.2f}")
