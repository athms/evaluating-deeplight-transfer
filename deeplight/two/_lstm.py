#!/usr/bin/python
import tensorflow as tf
from modules import Module


class LSTM(Module):
    """LSTM cell."""
    def __init__(
      self,
      dim=64,
      input_dim=None,
      batch_size=32,
      weights_init=tf.truncated_normal_initializer(stddev=0.01),
      bias_init=tf.constant_initializer(0.0),
      sequence_length=None,
      input_keep_prob=1.0,
      output_keep_prob=1.0,
      forget_bias=1.0,
      reverse_input=False,
      name="lstm"):
      """Basic LSTM cell implementation.

      Args:
          dim (int): Dim of hidden embedding.
          input_dim (int, optional): Size of input.
          batch_size (int, optional): Batch size. Defaults to 32.
          weights_init (weight initializer, optional): initializer for weights.
            Defaults to tf.truncated_normal_initializer(stddev=0.01).
          bias_init (weight initializer, optional): initializer for biases.
            Defaults to tf.constant_initializer(0.0).
          sequence_length (int, optional): Length of sequence. Defaults to None.
          input_keep_prob (float, optional): keep_prob for dropout applied to LSTM input.
            Defaults to 1.0.
          output_keep_prob (float, optional): keep_prob for dropout applied to LSTM output.
            Defaults to 1.0.
          forget_bias (float, optional): Bias of forget gate. Defaults to 1.0.
          reverse_input (bool, optional): Whether LSTM iterates through sequence 
              from beginning-to_end or end-to-beginning. Defaults to False.
          name (str, optional): Name assigned to layer. Defaults to "lstm".
      """
      self.name = name
      Module.__init__(self)

      with tf.name_scope(self.name):

        self.input_dim = input_dim  # n features
        self.dim = dim  # n output features
        self.seq_length = sequence_length

        self.batch_size = batch_size
        self.input_keep_prob = input_keep_prob
        self.output_keep_prob = output_keep_prob
        self.forget_bias = forget_bias
        self.reverse_input = reverse_input

        def _dropout_check_false():
            return tf.constant(1.0)
        def _input_dropout_check_true():
            return tf.cast(tf.multiply(self.input_keep_prob, 1), dtype=tf.float32)
        _input_dropout_check = self.input_keep_prob <= tf.constant(1.0)
        self.input_keep_prob_tensor = tf.cond(
          _input_dropout_check,
          _input_dropout_check_true,
          _dropout_check_false
        )
        def _output_dropout_check_true():
            return tf.cast(tf.multiply(self.output_keep_prob, 1), dtype=tf.float32)
        output_dropout_check = self.output_keep_prob <= tf.constant(1.0)
        self.output_keep_prob_tensor = tf.cond(
          output_dropout_check,
          _output_dropout_check_true,
          _dropout_check_false
        )

        self.Wxh = tf.get_variable(
          self.name+'/weights_xh',
          shape=[4 * self.dim, self.input_dim],
          initializer=weights_init
        )
        self.bxh = tf.get_variable(
          self.name+'/biases_xh',
          shape=[4 * self.dim],
          initializer=bias_init
        )
        self.Whh = tf.get_variable(
          self.name+'/weights_hh',
          shape=[4 * self.dim, self.dim],
          initializer=weights_init
        )
        self.bhh = tf.get_variable(
          self.name+'/biases_hh',
          shape=[4 * self.dim],
          initializer=bias_init
        )
        self.W = tf.concat([self.Wxh, self.Whh], axis=1)

        # i, f, o gate weight indices
        self.i_idx = tf.range(0, self.dim)
        self.j_idx = tf.range(self.dim, 2*self.dim)
        self.f_idx = tf.range(2*self.dim, 3*self.dim)
        self.o_idx = tf.range(3*self.dim, 4*self.dim)

      self._init_cell()


    def _init_cell(self):
      """Initialze cell."""
      self.h = [
        tf.zeros((self.dim, self.batch_size)) for _ in range(self.seq_length+1)
      ]
      self.c = [
        tf.zeros((self.dim, self.batch_size)) for _ in range(self.seq_length+1)
      ]
      self.h_out = [
        tf.zeros((self.batch_size, 1, self.dim*2)) for _ in range(self.seq_length)
      ]

      self.gates_xh = [
        tf.zeros((self.dim*4, self.batch_size)) for _ in range(self.seq_length)
      ]
      self.gates_hh = [
        tf.zeros((self.dim*4, self.batch_size)) for _ in range(self.seq_length)
      ]
      self.gates_pre = [
        tf.zeros((self.dim*4, self.batch_size)) for _ in range(self.seq_length)
      ]
      self.gates = [
        tf.zeros((self.dim*4, self.batch_size)) for _ in range(self.seq_length)
      ]


    def forward(self, x):
      """Forward pass

      Args:
          x (Tensor): Input

      Returns:
          Tensor: Hidden LSTM state of last sequence step.
      """
      with tf.variable_scope(self.name, reuse=True):

        self._init_cell()

        if len(x.get_shape().as_list()) != 3:
          self.x = tf.reshape(x, [-1, self.seq_length, self.input_dim])
        else:
          self.x = x

        self.x_time_first = tf.transpose(tf.transpose(self.x, [1, 0, 2]), [0, 2, 1])
        if self.reverse_input:
          self.x_time_first = tf.reverse(self.x_time_first, [0])

        for t in range(self.seq_length):

          x_t = tf.nn.dropout(
            tf.gather(self.x_time_first, t),
            keep_prob=self.input_keep_prob_tensor
          )

          self.gates_xh[t] = tf.matmul( self.Wxh, x_t) + tf.expand_dims(self.bxh, 1)
          self.gates_hh[t] = tf.matmul( self.Whh, self.h[t-1]) + tf.expand_dims(self.bhh, 1)
          self.gates_pre[t] = self.gates_xh[t] + self.gates_hh[t]

          i_gate = tf.sigmoid(tf.gather(self.gates_pre[t], self.i_idx))
          j_gate = tf.tanh(tf.gather(self.gates_pre[t], self.j_idx))
          f_gate = tf.sigmoid(tf.gather(self.gates_pre[t], self.f_idx) + self.forget_bias)
          o_gate = tf.sigmoid(tf.gather(self.gates_pre[t], self.o_idx))

          self.c[t] = f_gate*self.c[t-1] + i_gate*j_gate
          self.h[t] = o_gate*tf.tanh(self.c[t])
          self.gates[t] = tf.concat([i_gate, j_gate, f_gate, o_gate], axis=0)

          h_out = tf.transpose(
            tf.nn.dropout(self.h[t], keep_prob=self.output_keep_prob_tensor)
          )

        return h_out


    def _epsilon_lrp(self, R, epsilon=1e-3):
      """Epsilon LRP"""
      self.epsilon = epsilon
      self.propagation_rule = self.__epsilon_lrp
      return self._lrp(R)


    def __epsilon_lrp(
      self,
      hin,
      weights,
      biases,
      hout,
      Rout,
      n_bias_units,
      bias_factor=0):
      """Epsilon LRP implementation"""
      sign_out = tf.where(
        tf.greater_equal(hout, 0.),
        tf.ones_like(hout, dtype=tf.float32),
        1.*tf.ones_like(hout, dtype=tf.float32)
      )
      sign_out = tf.expand_dims(sign_out, 0)

      numer = tf.expand_dims(weights, -1) * tf.expand_dims(hin, 1)
      denom = tf.expand_dims(hout, 0) + self.epsilon*sign_out*1.
      message = numer/denom * tf.expand_dims(Rout, 0)
      return tf.reduce_sum(message, axis=1)


    def clean(self):
      """Clean cell states and weights."""
      self.Rx = None
      self.Rh = None
      self.Rc = None
      self.Rg = None
      self.R = None
      self.h = None
      self.c = None
      self.h_out = None
      self.gates_xh = None
      self.gates_hh = None
      self.gates_pre = None
      self.gates = None

    def _lrp(self, R):
      """LRP implementation."""
      self.Rx = [tf.zeros((self.dim, self.batch_size))] * self.seq_length
      self.Rh = [tf.zeros((self.dim, self.batch_size))] * (self.seq_length+1)
      self.Rc = [tf.zeros((self.dim, self.batch_size))] * (self.seq_length+1)
      self.Rg = [tf.zeros((self.dim, self.batch_size))] * (self.seq_length)

      self.R = tf.transpose(R)

      if self.reverse_input:
        self.Rh[self.seq_length-1] = tf.gather(self.R, tf.range(self.dim, 2*self.dim))
      else:
        self.Rh[self.seq_length-1] = tf.gather(self.R, tf.range(0, self.dim))

      for t in reversed(range(self.seq_length)):

        x_t = tf.gather(self.x_time_first, t)

        i_gate = tf.gather(self.gates[t], self.i_idx)
        j_gate = tf.gather(self.gates[t], self.j_idx)
        f_gate = tf.gather(self.gates[t], self.f_idx)

        j_gate_pre = tf.gather(self.gates_pre[t], self.j_idx)
        j_gate_pre = tf.gather(self.gates_pre[t], self.j_idx)

        self.Rc[t] += self.Rh[t]
        self.Rc[t-1] = self.propagation_rule(
          f_gate * self.c[t-1],
          tf.eye((self.dim)),
          tf.zeros((self.dim)),
          self.c[t],
          self.Rc[t],
          2*self.dim
        )
        self.Rg[t] = self.propagation_rule(
          i_gate * j_gate,
          tf.eye((self.dim)),
          tf.zeros((self.dim)),
          self.c[t],
          self.Rc[t],
          2*self.dim
        )
        self.Rx[t] = self.propagation_rule(
          x_t,
          tf.transpose(tf.gather(self.Wxh, self.j_idx)),
          tf.gather(self.bxh, self.j_idx) + tf.gather(self.bhh, self.j_idx),
          j_gate_pre,
          self.Rg[t],
          self.dim + self.input_dim
        )
        self.Rh[t-1] = self.propagation_rule(
          self.h[t-1],
          tf.transpose(tf.gather(self.Whh, self.j_idx)),
          tf.gather(self.bxh, self.j_idx) + tf.gather(self.bhh, self.j_idx),
          j_gate_pre,
          self.Rg[t],
          self.dim + self.input_dim
        )

      Rout = []
      for sample in range(self.batch_size):
        if self.reverse_input:
          seq_iterator = reversed(range(self.seq_length))
        else:
          seq_iterator = range(self.seq_length)
        for t in seq_iterator:
          R = tf.gather(self.Rx[t], sample, axis=1)
          Rout.append(tf.expand_dims(R, 0))
      Rout = tf.concat(Rout, 0)

      return Rout


class LSTM_bidirectional(Module):
    """Bi-directional LSTM cell."""
    def __init__(
      self,
      dim,
      input_dim=None,
      batch_size=None,
      weights_init=tf.truncated_normal_initializer(stddev=0.01),
      bias_init=tf.constant_initializer(0.0),
      sequence_length=None,
      input_keep_prob=1.0,
      output_keep_prob=1.0,
      forget_bias=1.0,
      name="lstm_bidirectional"):
      """Basic bi-directional LSTM cell implementation;
      combining two LSTM cells.

      Args:
          dim (int): Dim of hidden embedding of each LSTM cell.
          input_dim (int, optional): Size of input.
          batch_size (int, optional): Batch size. Defaults to 32.
          weights_init (weight initializer, optional): initializer for weights.
            Defaults to tf.truncated_normal_initializer(stddev=0.01).
          bias_init (weight initializer, optional): initializer for biases.
            Defaults to tf.constant_initializer(0.0).
          sequence_length (int, optional): Length of sequence. Defaults to None.
          input_keep_prob (float, optional): keep_prob for dropout applied to LSTM input.
            Defaults to 1.0.
          output_keep_prob (float, optional): keep_prob for dropout applied to LSTM output.
            Defaults to 1.0.
          forget_bias (float, optional): Bias of forget gate. Defaults to 1.0.
          reverse_input (bool, optional): Whether LSTM iterates through sequence 
              from beginning-to_end or end-to-beginning. Defaults to False.
          name (str, optional): Name assigned to layer. Defaults to "lstm_bidirectional".
      """
      self.name = name
      Module.__init__(self)

      with tf.name_scope(self.name):

        self.input_dim = input_dim  # n features
        self.dim = dim  # dim of embedding
        self.seq_length = sequence_length

        self.batch_size = batch_size
        self.input_keep_prob = input_keep_prob
        self.output_keep_prob = output_keep_prob
        self.forget_bias = forget_bias

        self.fw_cell = LSTM(
          dim,
          batch_size=self.batch_size,
          input_dim=self.input_dim,
          weights_init=weights_init,
          bias_init=bias_init,
          sequence_length=self.seq_length,
          input_keep_prob=self.input_keep_prob,
          output_keep_prob=self.output_keep_prob,
          forget_bias=self.forget_bias,
          name="lstm_fw"
        )

        self.bw_cell = LSTM(
          dim,
          batch_size=self.batch_size,
          input_dim=self.input_dim,
          weights_init=weights_init,
          bias_init=bias_init,
          sequence_length=self.seq_length,
          input_keep_prob=self.input_keep_prob,
          output_keep_prob=self.output_keep_prob,
          forget_bias=self.forget_bias,
          reverse_input=True,
          name="lstm_bw"
        )


    def forward(self, x):
      """Forward pass.

      Args:
          x (Tensor): Input tensor

      Returns:
          Tensor: Concatenated hidden state of both LSTM
            units for last sequence step.
      """
      with tf.variable_scope(self.name, reuse=True):
        h_fw = self.fw_cell.forward(x)
        h_bw = self.bw_cell.forward(x)
        return tf.concat([h_fw, h_bw], axis=1)


    def _epsilon_lrp(self, R, epsilon=1e-3):
      """Epsilon LRP"""
      R_fw = self.fw_cell._epsilon_lrp(R, epsilon=epsilon)
      R_bw = self.bw_cell._epsilon_lrp(R, epsilon=epsilon)
      return R_fw + R_bw


    def clean(self):
      """Clean cell states and weights."""
      self.fw_cell.clean()
      self.bw_cell.clean()
      self.R = None
