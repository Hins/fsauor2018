# -*- coding: utf-8 -*-
# ========================================
# Author: Xueyou Luo 
# Email: xueyou.luo@aidigger.com 
# Copyright: Eigen Tech @ 2018 
# ========================================

import argparse
import json
import time

import numpy as np
import tensorflow as tf

from dataset import DataSet
from model import Model
from utils import *
from tensorflow.python.framework import graph_util

def add_arguments(parser):
    """Build ArgumentParser."""
    parser.register("type", "bool", lambda v: v.lower() == "true")

    # mode
    parser.add_argument("--mode", type=str, default='train', help="running mode: train | eval | inference")

    # data
    parser.add_argument("--data_files", type=str, nargs='+', default=None, help="data file for train or inference")
    parser.add_argument("--eval_files", type=str, nargs='+', default=None, help="eval data file for evaluation")
    parser.add_argument("--label_file", type=str, default=None, help="label file")
    parser.add_argument("--vocab_file", type=str, default=None, help="vocab file")
    parser.add_argument("--embed_file", type=str, default=None, help="embedding file to restore")
    parser.add_argument("--out_file", type=str, default=None, help="output file for inference")
    parser.add_argument("--split_word", type='bool', nargs="?", const=True, default=True, help="Whether to split word when oov")
    parser.add_argument("--max_len", type=int, default=20, help='max length for doc')
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--reverse", type='bool', nargs="?", const=True, default=False, help="Whether to reverse data")
    parser.add_argument("--prob", type='bool', nargs="?", const=True, default=False, help="Whether to export prob")

    # model
    parser.add_argument("--num_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--decay_schema", type=str, default='hand', help = 'learning rate decay: exp | hand')
    parser.add_argument("--encoder", type=str, default='gnmt', help="gnmt | elmo")
    parser.add_argument("--decay_steps", type=int, default=10000, help="decay steps")
    parser.add_argument("--learning_rate", type=float, default=0.000001, help="Learning rate. RMS: 0.001 | 0.0001")
    parser.add_argument("--focal_loss", type=float, default=2., help="gamma of focal loss")
    parser.add_argument("--embedding_dropout", type=float, default=0.1, help="embedding_dropout")
    parser.add_argument("--max_gradient_norm", type=float, default=5.0, help="Clip gradients to this norm.")
    parser.add_argument("--dropout_keep_prob", type=float, default=0.8, help="drop out keep ratio for training")
    parser.add_argument("--weight_keep_drop", type=float, default=0.8, help="weight keep drop")
    parser.add_argument("--l2_loss_ratio", type=float, default=0.0, help="l2 loss ratio")
    parser.add_argument("--rnn_cell_name", type=str, default='lstm', help = 'rnn cell name')
    parser.add_argument("--embedding_size", type=int, default=300, help="embedding_size")
    parser.add_argument("--num_units", type=int, default=300, help="num_units")
    parser.add_argument("--double_decoder", type='bool', nargs="?", const=True, default=False, help="Whether to double decoder size")
    parser.add_argument("--variational_dropout", type='bool', nargs="?", const=True, default=True, help="Whether to use variational_dropout")

    # clf
    parser.add_argument("--target_label_num", type=int, default=2, help="target_label_num")
    parser.add_argument("--feature_num", type=int, default=1, help="feature_num")

    # train
    parser.add_argument("--need_early_stop", type='bool', nargs="?", const=True, default=False, help="Whether to early stop")
    parser.add_argument("--patient", type=int, default=5, help="patient of early stop")
    parser.add_argument("--debug", type='bool', nargs="?", const=True, default=False, help="Whether use debug mode")
    parser.add_argument("--num_train_epoch", type=int, default=1, help="training epoches")
    parser.add_argument("--steps_per_stats", type=int, default=10, help="steps to print stats")
    parser.add_argument("--steps_per_summary", type=int, default=30, help="steps to save summary")
    parser.add_argument("--steps_per_eval", type=int, default=10, help="steps to save model")

    parser.add_argument("--checkpoint_dir", type=str, default='/tmp/visual-semantic', help="checkpoint dir to save model")
    #cnn
    parser.add_argument("--num_filters", type=int, default=200, help="Number of filters per filter size")

def convert_to_hparams(params):
    hparams = tf.contrib.training.HParams()
    for k,v in params.items():
        hparams.add_hparam(k,v)
    return hparams

def inference(flags):
    from collections import defaultdict
    print_out("inference data file {0}".format(flags.data_files))
    dataset = DataSet(flags.data_files, flags.vocab_file, flags.label_file, 5*flags.batch_size, reverse=flags.reverse, split_word=flags.split_word, max_len=flags.max_len)
    hparams = load_hparams(flags.checkpoint_dir,{"mode":'inference','checkpoint_dir':flags.checkpoint_dir+"/best_eval"})
    with tf.Session(config = get_config_proto(log_device_placement=False)) as sess:
        model = Model(hparams)
        model.build()
        
        try:
            model.restore_model(sess)  #restore best solution
        except Exception as e:
            print("unable to restore model with exception",e)
            exit(1)

        scalars = model.scalars.eval(session=sess)
        print("Scalars:", scalars)
        weight = model.weight.eval(session=sess)
        print("Weight:",weight)
        cnt = 0
        checkpoint_loss, acc = 0.0, 0.0 
        predicts, truths = defaultdict(list), defaultdict(list)
        for i,(source, lengths, targets, _) in enumerate(dataset.get_next(shuffle=False)):
            batch_loss, accuracy, batch_size, predict = model.eval_clf_one_step(sess, source, lengths,targets)
            for i,p in enumerate(predict):
                for j in range(flags.feature_num):
                    label_name = dataset.i2l[j]
                    truths[label_name].append(targets[i][j])
                    predicts[label_name].append(p[j])
                    
                    tag = dataset.tag_i2l[np.argmax(p[j])]
                    dataset.items[cnt + i][label_name] = tag
            cnt += len(lengths)
            print_out("\r# process {0:.2%}".format(cnt / dataset.data_size), new_line=False)

            checkpoint_loss += batch_loss
            acc += accuracy
            if (i + 1) % 100 == 0:
                print_out("# batch %d/%d" % (i + 1, dataset.num_batches))
    
    results = {}
    total_f1 = 0.0
    for label_name in dataset.label_names:
        # print("# Get f1 score for",label_name)
        f1,precision,recall = cal_f1(model.hparams.target_label_num,np.asarray(predicts[label_name]),np.asarray(truths[label_name]))
        results[label_name] = f1
        total_f1 += f1
        print_out("# {0} - f1:{1} - pre:{2}  - rec:{3}".format(label_name, f1, precision,recall))
    final_f1 = total_f1 / len(results)
    print_out("# Eval loss %.5f, f1 %.5f" % (checkpoint_loss / i, final_f1))
    print_out("# Done")

    
    print_out("# Write result to file ...")
    with open(flags.out_file, 'w') as f:
        for item in dataset.items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
     
    ''' 
    results = {}
    total_f1 = 0.0
    total_pre= 0.0
    total_recall= 0.0
    emotion_names = ['无', '喜好', '悲伤', '愤怒', '高兴', '厌恶', '惊讶', '恐惧']
    with open('scripts/weibo/result.txt','w',encoding='utf-8') as f:
        for label_name in dataset.label_names:
            # print("# Get f1 score for",label_name)
            precision, recall, f1, total_f1,total_pre,total_recall = actual_f1(model.hparams.target_label_num,np.asarray(predicts[label_name]),
                                                     np.asarray(truths[label_name]))
            f.write("# Eval loss %.5f, f1 %.5f,pre %.5f,recall %.5f " % (checkpoint_loss / i, total_f1,total_pre,total_recall))
            id = 0
            for emotion in emotion_names:
                f.write("# emotion:{0} - f1:{1} - pre:{2} - recall{3}".format(emotion, f1[id],precision[id],recall[id]))
                print("# emotion:{0} - f1:{1} - pre:{2} - recall{3}".format(emotion, f1[id],precision[id],recall[id]))
                id += 1
            results[label_name] = f1        
    final_f1 = total_f1 / len(results)
    print_out("# Eval loss %.5f, f1 %.5f,pre %.5f,recall %.5f " % (checkpoint_loss / i, final_f1,total_pre,total_recall))
    print_out("# Done")
    '''

def test(flags,content):
    from collections import defaultdict
    dataset = DataSet(content, flags.vocab_file, flags.label_file,  reverse=flags.reverse, split_word=flags.split_word, max_len=flags.max_len)
    hparams = load_hparams(flags.checkpoint_dir,
                           {"mode": 'inference', 'checkpoint_dir': flags.checkpoint_dir + "/best_eval",
                            'embed_file': 'scripts/data/embedding.txt'})
    with tf.Session(config = get_config_proto(log_device_placement=False)) as sess:
        model = Model(hparams)
        model.build()
        try:
            model.restore_model(sess)
        except Exception as e:
            print("unable to restore model with exception",e)
            exit(1)
        scalars = model.scalars.eval(session=sess)
        weight = model.weight.eval(session=sess)
        cnt = 0
    source ,length,target,id = dataset.process_sentence()
    predict,logit = model.inference_clf_one_batch(sess, source, length)
    for i, p in enumerate(predict):
        for j in range(flags.feature_num):
            label_name = dataset.i2l[j]
            tag = dataset.tag_i2l[np.argmax(p[j])]
    return tag

def train_eval_clf(model, sess, dataset):
    from collections import defaultdict
    checkpoint_loss, acc = 0.0, 0.0

    predicts, truths = defaultdict(list), defaultdict(list)
    for i,(source, lengths, targets, _) in enumerate(dataset.get_next(shuffle=False)):
        batch_loss, accuracy, batch_size, predict = model.eval_clf_one_step(sess, source, lengths, targets)
        # batch * 20 * 4
        for i,p in enumerate(predict):
            for j in range(model.hparams.feature_num):
                label_name = dataset.i2l[j]
                truths[label_name].append(targets[i][j])
                predicts[label_name].append(p[j])
        checkpoint_loss += batch_loss
        acc += accuracy
        if (i+1) % 100 == 0:
            print_out("# batch %d/%d" %(i+1,dataset.num_batches))

    precision_list = {}
    results = {}
    total_f1 = 0.0
    total_precision = 0.0
    for label_name in dataset.label_names:
        # print("# Get f1 score for",label_name)
        f1,precision,recall = cal_f1(model.hparams.target_label_num,np.asarray(predicts[label_name]),np.asarray(truths[label_name]))
        results[label_name] = f1
        precision_list[label_name] = precision
        total_f1 += f1
        total_precision += precision
        print("# {0} - {1}".format(label_name,f1))

    final_f1 = total_f1 / len(results)
    avg_precision = total_precision / len(precision_list)
        
    print_out( "# Eval loss %.5f, f1 %.5f" % (checkpoint_loss/i, final_f1))
    return -1 * final_f1, avg_precision, checkpoint_loss/i

def train_clf(flags):
    dataset = DataSet(flags.data_files, flags.vocab_file, flags.label_file, flags.batch_size, reverse=flags.reverse, split_word=flags.split_word, max_len=flags.max_len)
    eval_dataset = DataSet(flags.eval_files, flags.vocab_file, flags.label_file, 5 * flags.batch_size, reverse=flags.reverse, split_word=flags.split_word, max_len=flags.max_len)

    params = vars(flags)
    params['vocab_size'] = len(dataset.w2i)
    hparams = convert_to_hparams(params)

    save_hparams(flags.checkpoint_dir, hparams)
    print(hparams)

    train_graph = tf.Graph()
    eval_graph = tf.Graph()

    with train_graph.as_default():
        train_model = Model(hparams)
        train_model.build()
        initializer = tf.global_variables_initializer()

    with eval_graph.as_default():
        eval_hparams = load_hparams(flags.checkpoint_dir,{"mode":'eval','checkpoint_dir':flags.checkpoint_dir+"/best_eval"})
        eval_model = Model(eval_hparams)
        eval_model.build()

    train_sess = tf.Session(graph=train_graph, config=get_config_proto(log_device_placement=False ))
    train_model.init_model(train_sess, initializer=initializer)
    try:
        train_model.restore_model(train_sess)
    except:
        print_out("unable to restore model, train from scratch")
            
    print_out("# Start to train with learning rate {0}, {1}".format(flags.learning_rate,time.ctime()))

    global_step = train_sess.run(train_model.global_step)
    print("# Global step", global_step)

    eval_ppls = []
    best_eval = 1000000000
    pre_best_checkpoint = None
    final_learn = 2

    for epoch in range(flags.num_train_epoch):
        step_time, checkpoint_loss, acc, total_iter, train_acc, total_train_acc, total_train_loss, eval_acc, total_eval_loss = 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0
        for i,(source, lengths, targets, _) in enumerate(dataset.get_next()):
            global_step = i
            if lengths[0] < 6 and flags.encoder == "cnn":
                continue
            total_iter = i + 1
            start_time = time.time()
            add_summary = (global_step % flags.steps_per_summary == 0)
            # batch_loss, global_step, accuracy, token_num, batch_size = train_model.train_clf_one_step(train_sess,source, lengths, targets, add_summary=add_summary, run_info=add_summary and flags.debug)
            batch_loss, accuracy = train_model.train_clf_one_step(train_sess,
                                                                    source,
                                                                    lengths,
                                                                    targets,
                                                                    add_summary=add_summary,
                                                                    run_info=add_summary and flags.debug)
            print("Epoch = {0}, iteration = {1}, train accu = {2}".format(epoch, i, accuracy))
            step_time += (time.time() - start_time)
            checkpoint_loss += batch_loss
            total_train_loss += batch_loss
            train_acc += accuracy
            total_train_acc += accuracy
            acc += accuracy

            if global_step == 0:
                continue

            '''
            if global_step % flags.steps_per_stats == 0:
                train_acc = (acc / flags.steps_per_stats) * 100
                acc_summary = tf.Summary()
                acc_summary.value.add(tag='accuracy', simple_value = train_acc)
                train_model.summary_writer.add_summary(acc_summary, global_step=global_step)

                print_out(
                    "# Epoch %d  global step %d loss %.5f batch %d/%d lr %g "
                    "accuracy %.5f wps %.2f step time %.2fs" %
                    (epoch+1, global_step, checkpoint_loss/flags.steps_per_stats, i+1,dataset.num_batches, train_model.learning_rate.eval(session=train_sess),
                    train_acc, (iters)/step_time, step_time/(flags.steps_per_stats)))
                step_time, checkpoint_loss, iters, acc = 0.0, 0.0, 0, 0.0
            '''

            if global_step % flags.steps_per_eval == 0:
                print_out("# global step {0}, eval model at {1}".format(global_step, time.ctime()))
                checkpoint_path = train_model.save_model(train_sess)
                print("checkpoint_path is {0}".format(checkpoint_path))
                with tf.Session(graph=eval_graph, config=get_config_proto(log_device_placement=False)) as eval_sess:
                    eval_model.init_model(eval_sess)
                    eval_model.restore_ema_model(eval_sess, checkpoint_path)
                    eval_ppl, avg_prec, eval_loss = train_eval_clf(eval_model, eval_sess, eval_dataset)
                    eval_acc += avg_prec
                    total_eval_loss += eval_loss
                    print_out("# current result {0}, previous best result {1}".format(eval_ppl,best_eval))

                    loss_summary = tf.Summary()
                    loss_summary.value.add(tag='eval_loss', simple_value = eval_loss)
                    train_model.summary_writer.add_summary(loss_summary, global_step=global_step)
                    if eval_ppl < best_eval:
                        pre_best_checkpoint = checkpoint_path
                        eval_model.save_model(eval_sess,global_step)
                        best_eval = eval_ppl
                    eval_ppls.append(eval_ppl)
                if flags.need_early_stop:
                    if early_stop(eval_ppls, flags.patient):
                        print_out("# No loss decrease, restore previous best model and set learning rate to half of previous one")
                        current_lr = train_model.learning_rate.eval(session=train_sess)
                        if final_learn > 0:
                            final_learn -= 1
                        else:
                            print_out("# Early stop, exit")
                            exit(0)
                        train_model.saver.restore(train_sess, pre_best_checkpoint)
                        lr = tf.assign(train_model.learning_rate, current_lr/10)
                        if final_learn==0:
                            dropout = tf.assign(train_model.dropout_keep_prob, 1.0)
                            emd_drop = tf.assign(train_model.embedding_dropout, 0.0)
                            train_sess.run([dropout,emd_drop])
                        train_sess.run(lr)
                        eval_ppls = [best_eval]
                        continue
        if flags.encoder == "bi-lstm":
            source_tokens = train_sess.graph.get_operation_by_name("source_tokens").outputs[0]
            sequence_length = train_sess.graph.get_operation_by_name("sequence_length").outputs[0]
            final_logits = train_sess.graph.get_operation_by_name("lstm-output/final_logits").outputs[0]
            tf.saved_model.simple_save(train_sess,
                                       flags.checkpoint_dir + str(epoch) + "/model",
                                       inputs={"source_tokens": source_tokens,
                                               "sequence_length": sequence_length},
                                       outputs={"final_logits": final_logits})
        elif flags.encoder == "cnn":
            source_tokens = train_sess.graph.get_operation_by_name("source_tokens").outputs[0]
            sequence_length = train_sess.graph.get_operation_by_name("sequence_length").outputs[0]
            final_logits = train_sess.graph.get_operation_by_name("output/scores").outputs[0]
            tf.saved_model.simple_save(train_sess,
                                       flags.checkpoint_dir + str(epoch) + "/model",
                                       inputs={"source_tokens": source_tokens,
                                               "sequence_length": sequence_length},
                                       outputs={"final_logits": final_logits})
        print_out("# Finsh epoch {1}, global step {0}, train acc {2}, train loss {3}, eval acc {4}, eval loss {5}".format(
            global_step, epoch+1, total_train_acc/total_iter, total_train_loss/total_iter, eval_acc/total_iter, total_eval_loss/total_iter))
    print_out("# Best accuracy {0}".format(best_eval))

def test_result(content):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    flags, unparsed = parser.parse_known_args()

    logit = test(flags, content)
    return str(logit)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    flags, unparsed = parser.parse_known_args()
    if flags.mode == 'train':
        train_clf(flags)
    elif flags.mode == 'inference':
        inference(flags)
