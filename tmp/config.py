import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_integer('word_embedding_size', 256, 'word embedding size')
flags.DEFINE_integer('position_embedding_size', 100, 'position embedding size')
flags.DEFINE_integer('embedding_window', 4, 'embedding window')
flags.DEFINE_integer('hidden_size', 128, 'word2vec weight size')
flags.DEFINE_float('train_set_ratio', 0.3, 'train set ratio')
flags.DEFINE_integer('batch_size', 256, 'train batch size')
flags.DEFINE_integer('top_k_sim', -5, 'top k similarity items')

flags.DEFINE_float('stddev', 0.01, 'stddev for W initializer')
flags.DEFINE_integer('negative_sample_size', 5, 'negative sample size')
flags.DEFINE_integer('max_len_size', 10, 'maximum length size')
flags.DEFINE_integer('train_max_len_size', 100, 'maximum length size in training set')
flags.DEFINE_integer('MAX_GRAD_NORM', 5, 'maximum gradient norm')
flags.DEFINE_integer('epoch_size', 10, 'epoch size')

flags.DEFINE_string('summaries_dir', './tb/', 'Summaries directory')
flags.DEFINE_string('train_summary_writer_path', 'train/', 'train summaries directory')
flags.DEFINE_string('position_train_summary_writer_path', '/position_train', 'position train summary writer path')
flags.DEFINE_string('position_test_summary_writer_path', '/position_test', 'position test summary writer path')
flags.DEFINE_string('sg_train_summary_writer_path', '/sg_train', 'skip-gram train summary writer path')
flags.DEFINE_string('sg_test_summary_writer_path', '/sg_test', 'skip-gram test summary writer path')

cfg = tf.app.flags.FLAGS