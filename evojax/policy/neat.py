# Copyright 2024 The EvoJAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
from typing import Optional

import jax
import jax.numpy as jnp

from evojax.policy.base import PolicyNetwork
from evojax.policy.base import PolicyState
from evojax.task.base import TaskState
from evojax.util import create_logger


class NEATPolicy(PolicyNetwork):
    """Policy that decodes NEAT genomes emitted by ``evojax.algo.neat.NEAT``."""

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 max_hidden_nodes: int,
                 propagation_steps: Optional[int] = None,
                 logger: Optional[logging.Logger] = None):
        if logger is None:
            self._logger = create_logger(name='NEATPolicy')
        else:
            self._logger = logger

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.max_hidden_nodes = int(max_hidden_nodes)

        if self.input_dim < 0 or self.output_dim <= 0 or self.max_hidden_nodes < 0:
            raise ValueError('Invalid NEATPolicy dimensions.')

        self._bias_nodes = 1
        self._max_nodes = (
            self._bias_nodes + self.input_dim + self.output_dim + self.max_hidden_nodes
        )

        weight_size = self._max_nodes * self._max_nodes
        self._weights_slice = slice(0, weight_size)
        self._state_slice = slice(2 * weight_size, 3 * weight_size)
        node_type_start = 3 * weight_size
        self._node_type_slice = slice(node_type_start, node_type_start + self._max_nodes)
        self._node_act_slice = slice(self._node_type_slice.stop,
                                     self._node_type_slice.stop + self._max_nodes)
        self.num_params = self._node_act_slice.stop
        self._logger.info('NEATPolicy.num_params = %d', self.num_params)

        self._bias_index = 0
        self._input_slice = slice(self._bias_nodes,
                                  self._bias_nodes + self.input_dim)
        self._output_slice = slice(self._bias_nodes + self.input_dim,
                                   self._bias_nodes + self.input_dim + self.output_dim)

        self._prop_steps = propagation_steps or self._max_nodes

        def apply_activation(act_ids: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
            """Evaluates PrettyNEAT activation functions by id."""
            act_ids = act_ids.astype(jnp.int32)
            values = values.astype(jnp.float32)

            result = values
            result = jnp.where(act_ids == 2,
                               jnp.where(values > 0.0, 1.0, 0.0),
                               result)
            result = jnp.where(act_ids == 3,
                               jnp.sin(jnp.pi * values),
                               result)
            result = jnp.where(act_ids == 4,
                               jnp.exp(-0.5 * values * values),
                               result)
            result = jnp.where(act_ids == 5,
                               jnp.tanh(values),
                               result)
            result = jnp.where(act_ids == 6,
                               0.5 * (jnp.tanh(values / 2.0) + 1.0),
                               result)
            result = jnp.where(act_ids == 7,
                               -values,
                               result)
            result = jnp.where(act_ids == 8,
                               jnp.abs(values),
                               result)
            result = jnp.where(act_ids == 9,
                               jnp.maximum(0.0, values),
                               result)
            result = jnp.where(act_ids == 10,
                               jnp.cos(jnp.pi * values),
                               result)
            result = jnp.where(act_ids == 11,
                               values * values,
                               result)
            return result

        def split_params(params_vec: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            weights = params_vec[self._weights_slice].reshape(self._max_nodes, self._max_nodes)
            state = params_vec[self._state_slice].reshape(self._max_nodes, self._max_nodes)
            node_types = params_vec[self._node_type_slice]
            node_act = params_vec[self._node_act_slice]
            enabled = state > 0.0
            weights = jnp.where(enabled, weights, 0.0)
            node_types = jnp.round(node_types).astype(jnp.int32)
            node_act = jnp.round(node_act).astype(jnp.int32)
            return weights.astype(jnp.float32), node_types, node_act

        def forward_single(params_vec: jnp.ndarray,
                           obs_vec: jnp.ndarray) -> jnp.ndarray:
            weights, node_types, node_act = split_params(params_vec)
            obs_vec = obs_vec.astype(jnp.float32)
            activations = jnp.zeros(self._max_nodes, dtype=jnp.float32)
            activations = activations.at[self._bias_index].set(1.0)
            activations = activations.at[self._input_slice].set(obs_vec[:self.input_dim])

            active_mask = node_types > 0

            def body(_, current):
                summed = current @ weights
                updated = apply_activation(node_act, summed)
                updated = jnp.where(active_mask, updated, 0.0)
                updated = updated.at[self._bias_index].set(1.0)
                updated = updated.at[self._input_slice].set(obs_vec[:self.input_dim])
                return updated

            activations = jax.lax.fori_loop(
                0, self._prop_steps, lambda i, carry: body(i, carry), activations)
            return activations[self._output_slice]

        self._forward_fn = jax.jit(jax.vmap(
            forward_single,
            in_axes=(0, 0),
        ))

    def get_actions(self,
                    t_states: TaskState,
                    params: jnp.ndarray,
                    p_states: PolicyState) -> tuple[jnp.ndarray, PolicyState]:
        actions = self._forward_fn(params, t_states.obs)
        return actions, p_states
