import os
import tensorflow as tf
from model import CharRNN
from load_text import TextConvert

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', False, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_string('convert_path', 'model/shakespeare/text_convert.pkl', 
                                       'model/name/text_convert.pkl')
tf.flags.DEFINE_string('checkpoint_path', 'model/shakespeare/', 'checkpoint_path')
tf.flags.DEFINE_string('start_string', ' ', 'use this string to start generating')
tf.flags.DEFINE_integer('max_length', 1000, 'max length to generate')

def main(_):
    FLAGS.start_string = FLAGS.start_string
    convert = TextConvert(fname=FLAGS.convert_path)
    if os.path.isdir(FLAGS.checkpoint_path):
        FLAGS.checkpoint_path =  tf.train.latest_checkpoint(FLAGS.checkpoint_path)
           
    model = CharRNN(convert.vocab_size, sampling=True,
                    lstm_size=FLAGS.lstm_size, num_layers=FLAGS.num_layers,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size)

    model.load(FLAGS.checkpoint_path)
    start = convert.text2arr(FLAGS.start_string)
    
    arr = model.sample(FLAGS.max_length, start, convert.vocab_size)
    res = convert.arr2text(arr)
    print('get result: \n', res)


if __name__ == '__main__':
    tf.app.run()
