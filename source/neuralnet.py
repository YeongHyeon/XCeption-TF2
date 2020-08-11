import os
import tensorflow as tf
import source.layers as lay

class CNN(object):

    def __init__(self, height, width, channel, num_class, ksize, learning_rate=1e-3, ckpt_dir='./Checkpoint'):

        print("\nInitializing Short-XCeption...")
        self.height, self.width, self.channel, self.num_class = height, width, channel, num_class
        self.ksize, self.learning_rate = ksize, learning_rate
        self.ckpt_dir = ckpt_dir

        self.customlayers = lay.Layers()
        self.model(tf.zeros([1, self.height, self.width, self.channel]), verbose=True)

        self.optimizer = tf.optimizers.Adam(self.learning_rate)

        self.summary_writer = tf.summary.create_file_writer(self.ckpt_dir)

    def step(self, x, y, iteration=0, train=False):

        with tf.GradientTape() as tape:
            logits = self.model(x, verbose=False)
            smce = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
            loss = tf.math.reduce_mean(smce)

        score = self.customlayers.softmax(logits)
        pred = tf.argmax(score, 1)
        correct_pred = tf.equal(pred, tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        if(train):
            gradients = tape.gradient(loss, self.customlayers.params_trainable)
            self.optimizer.apply_gradients(zip(gradients, self.customlayers.params_trainable))

            with self.summary_writer.as_default():
                tf.summary.scalar('ResNet/loss', loss, step=iteration)
                tf.summary.scalar('ResNet/accuracy', accuracy, step=iteration)

        return loss, accuracy, score

    def save_params(self):

        vars_to_save = {}
        for idx, name in enumerate(self.customlayers.name_bank):
            vars_to_save[self.customlayers.name_bank[idx]] = self.customlayers.params_trainable[idx]
        vars_to_save["optimizer"] = self.optimizer

        ckpt = tf.train.Checkpoint(**vars_to_save)
        ckptman = tf.train.CheckpointManager(ckpt, directory=self.ckpt_dir, max_to_keep=3)
        ckptman.save()

    def load_params(self):

        vars_to_load = {}
        for idx, name in enumerate(self.customlayers.name_bank):
            vars_to_load[self.customlayers.name_bank[idx]] = self.customlayers.params_trainable[idx]
        vars_to_load["optimizer"] = self.optimizer

        ckpt = tf.train.Checkpoint(**vars_to_load)
        latest_ckpt = tf.train.latest_checkpoint(self.ckpt_dir)
        status = ckpt.restore(latest_ckpt)
        status.expect_partial()

    def model(self, x, verbose=False):

        if(verbose): print("input", x.shape)

        base = 32*3
        conv1_1 = self.xception(x, \
            ksize=self.ksize, inchannel=self.channel, outchannel=base*(6**0), name="conv1_1", verbose=verbose)
        conv1_2 = self.xception(conv1_1, \
            ksize=self.ksize, inchannel=base*(6**0), outchannel=base*(6**0), name="conv1_2", verbose=verbose)
        conv1_pool = self.customlayers.maxpool(conv1_2, pool_size=2, stride_size=2)

        conv2_1 = self.xception(conv1_pool, \
            ksize=self.ksize, inchannel=base*(6**0), outchannel=base*(6**1), name="conv2_1", verbose=verbose)
        conv2_2 = self.xception(conv2_1, \
            ksize=self.ksize, inchannel=base*(6**1), outchannel=base*(6**1), name="conv2_2", verbose=verbose)
        conv2_pool = self.customlayers.maxpool(conv2_2, pool_size=2, stride_size=2)

        conv3_1 = self.xception(conv2_pool, \
            ksize=self.ksize, inchannel=base*(6**1), outchannel=512*3, name="conv3_1", verbose=verbose)
        conv3_2 = self.xception(conv3_1, \
            ksize=self.ksize, inchannel=512*3, outchannel=512*3, name="conv3_2", verbose=verbose)

        gap = tf.reduce_mean(conv3_2, axis=(1, 2))
        if(verbose):
            num_param_fe = self.customlayers.num_params
            print("gap", gap.shape)

        fc1 = self.customlayers.fullcon(gap, \
            self.customlayers.get_weight(vshape=[1536, self.num_class], name="fullcon1"))
        if(verbose):
            print("fullcon1", fc1.shape)
            print("\nNum Parameter")
            print("Feature Extractor : %d" %(num_param_fe))
            print("Classifier        : %d" %(self.customlayers.num_params - num_param_fe))
            print("Total             : %d" %(self.customlayers.num_params))

        return fc1

    def xception(self, input, ksize, inchannel, outchannel, name="", verbose=False):

        num_div = 6
        midchannel = (inchannel + outchannel//num_div) // 2
        convtmp_1 = self.customlayers.conv2d(input, \
            self.customlayers.get_weight(vshape=[1, 1, inchannel, midchannel], name="%s_1" %(name)), \
            stride_size=1, padding='SAME')
        convtmp_1bn = self.customlayers.batch_norm(convtmp_1, name="%s_1bn" %(name))
        convtmp_1act = self.customlayers.elu(convtmp_1bn)

        splits = []
        if(midchannel % num_div == 0):
            for i in range(num_div): splits.append(midchannel//num_div)
        else:
            for i in range(num_div-1): splits.append(midchannel//num_div)
            spfin = midchannel - (splits[-1] * (num_div-1))
            splits.append(spfin)
        branches = tf.split(convtmp_1act, \
            num_or_size_splits=splits, axis=3)

        output = None
        for idx, branch in enumerate(branches):
            convtmp_2 = self.customlayers.conv2d(branch, \
                self.customlayers.get_weight(vshape=[ksize, ksize, splits[idx], outchannel//num_div], name="%s_2_b%d" %(name, idx+1)), \
                stride_size=1, padding='SAME')
            convtmp_2bn = self.customlayers.batch_norm(convtmp_2, name="%s_b%d_2bn" %(name, idx+1))
            convtmp_2act = self.customlayers.elu(convtmp_2bn)

            if(output is None): output = convtmp_2act
            else: output = tf.concat([output, convtmp_2act], axis=3)

        if(verbose): print(name, output.shape)
        return output
