from collections import deque

import numpy as np

from .. import utils


__all__ = [
    "get_all_layers",
    "get_all_layers_old",
    "get_output",
    "get_output_shape",
    "get_all_params",
    "get_all_bias_params",
    "get_all_non_bias_params",
    "count_params",
    "get_all_param_values",
    "set_all_param_values",
]


def get_all_layers(layer, treat_as_input=None):
    """
    This function gathers all layers below one or more given :class:`Layer`
    instances, including the given layer(s). Its main use is to collect all
    layers of a network just given the output layer(s). The layers are
    guaranteed to be returned in a topological order: a layer in the result
    list is always preceded by all layers its input depends on.

    :usage:
        >>> from lasagne.layers import InputLayer, DenseLayer
        >>> l_in = InputLayer((100, 20))
        >>> l1 = DenseLayer(l_in, num_units=50)
        >>> get_all_layers(l1) == [l_in, l1]
        True
        >>> l2 = DenseLayer(l_in, num_units=10)
        >>> get_all_layers([l2, l1]) == [l_in, l2, l1]
        True
        >>> get_all_layers([l1, l2]) == [l_in, l1, l2]
        True
        >>> l3 = DenseLayer(l2, num_units=20)
        >>> get_all_layers(l3) == [l_in, l2, l3]
        True
        >>> get_all_layers(l3, treat_as_input=[l2]) == [l2, l3]
        True

    :parameters:
        - layer : Layer or list
            the :class:`Layer` instance for which to gather all layers feeding
            into it, or a list of :class:`Layer` instances.
        - treat_as_input : None or iterable
            an iterable of :class:`Layer` instances to treat as input layers
            with no layers feeding into them. They will show up in the result
            list, but their incoming layers will not be collected (unless they
            are required for other layers as well).

    :returns:
        - layers : list
            a list of :class:`Layer` instances feeding into the given
            instance(s) either directly or indirectly, and the given
            instance(s) themselves, in topological order.
    """
    import warnings
    warnings.warn("get_all_layers() has been changed to return layers in "
                  "topological order. The former implementation is still "
                  "available as get_all_layers_old(), but will be removed "
                  "before the first release of Lasagne. To ignore this "
                  "warning, use `warnings.filterwarnings('ignore', "
                  "'.*topo.*')`.")

    # We perform a depth-first search. We add a layer to the result list only
    # after adding all its incoming layers (if any) or when detecting a cycle.
    # We use a LIFO stack to avoid ever running into recursion depth limits.
    try:
        queue = deque(layer)
    except TypeError:
        queue = deque([layer])
    seen = set()
    done = set()
    result = []

    # If treat_as_input is given, we pretend we've already collected all their
    # incoming layers.
    if treat_as_input is not None:
        seen.update(treat_as_input)

    while queue:
        # Peek at the leftmost node in the queue.
        layer = queue[0]
        if layer is None:
            # Some node had an input_layer set to `None`. Just ignore it.
            queue.popleft()
        elif layer not in seen:
            # We haven't seen this node yet: Mark it and queue all incomings
            # to be processed first. If there are no incomings, the node will
            # be appended to the result list in the next iteration.
            seen.add(layer)
            if hasattr(layer, 'input_layers'):
                queue.extendleft(reversed(layer.input_layers))
            elif hasattr(layer, 'input_layer'):
                queue.appendleft(layer.input_layer)
        else:
            # We've been here before: Either we've finished all its incomings,
            # or we've detected a cycle. In both cases, we remove the layer
            # from the queue and append it to the result list.
            queue.popleft()
            if layer not in done:
                result.append(layer)
                done.add(layer)

    return result


def get_all_layers_old(layer):
    """
    Earlier implementation of `get_all_layers()` that does a breadth-first
    search. Kept here to ease converting old models that rely on the order
    of get_all_layers() or get_all_params(). Will be removed before the
    first release of Lasagne.
    """
    if isinstance(layer, (list, tuple)):
        layers = list(layer)
    else:
        layers = [layer]
    layers_to_expand = list(layers)
    while len(layers_to_expand) > 0:
        current_layer = layers_to_expand.pop(0)
        children = []

        if hasattr(current_layer, 'input_layers'):
            children = current_layer.input_layers
        elif hasattr(current_layer, 'input_layer'):
            children = [current_layer.input_layer]

        # filter the layers that have already been visited, and remove None
        # elements (for layers without incoming layers)
        children = [child for child in children
                    if child not in layers and
                    child is not None]
        layers_to_expand.extend(children)
        layers.extend(children)

    return layers


def get_output(layer_or_layers, inputs=None, **kwargs):
    """
    Computes the output of the network at one or more given layers.
    Optionally, you can define the input(s) to propagate through the network
    instead of using the input variable(s) associated with the network's
    input layer(s).

    :parameters:
        - layer_or_layers : Layer or list
            the :class:`Layer` instance for which to compute the output
            expressions, or a list of :class:`Layer` instances.
        - inputs : None, Theano expression, numpy array, or dict
            If None, uses the input variables associated with the
            :class:`InputLayer` instances.
            If a Theano expression, this defines the input for a single
            :class:`InputLayer` instance. Will throw a ValueError if there
            are multiple :class:`InputLayer` instances.
            If a numpy array, this will be wrapped as a Theano constant
            and used just like a Theano expression.
            If a dictionary, any :class:`Layer` instance (including the
            input layers) can be mapped to a Theano expression or numpy
            array to use instead of its regular output.

    :returns:
        - output : Theano expression or list
            the output of the given layer(s) for the given network input

    :note:
        Depending on your network architecture, `get_output([l1, l2])` may
        be crucially different from `[get_output(l1), get_output(l2)]`. Only
        the former ensures that the output expressions depend on the same
        intermediate expressions. For example, when `l1` and `l2` depend on
        a common dropout layer, the former will use the same dropout mask for
        both, while the latter will use two different dropout masks.
    """
    from .input import InputLayer
    from .base import MergeLayer
    # obtain topological ordering of all layers the output layer(s) depend on
    treat_as_input = inputs.keys() if isinstance(inputs, dict) else []
    all_layers = get_all_layers(layer_or_layers, treat_as_input)
    # initialize layer-to-expression mapping from all input layers
    all_outputs = dict((layer, layer.input_var)
                       for layer in all_layers
                       if isinstance(layer, InputLayer) and
                       layer not in treat_as_input)
    # update layer-to-expression mapping from given input(s), if any
    if isinstance(inputs, dict):
        all_outputs.update((layer, utils.as_theano_expression(expr))
                           for layer, expr in inputs.items())
    elif inputs is not None:
        if len(all_outputs) > 1:
            raise ValueError("get_output() was called with a single input "
                             "expression on a network with multiple input "
                             "layers. Please call it with a dictionary of "
                             "input expressions instead.")
        for input_layer in all_outputs:
            all_outputs[input_layer] = utils.as_theano_expression(inputs)
    # update layer-to-expression mapping by propagating the inputs
    for layer in all_layers:
        if layer not in all_outputs:
            try:
                if isinstance(layer, MergeLayer):
                    layer_inputs = [all_outputs[input_layer]
                                    for input_layer in layer.input_layers]
                else:
                    layer_inputs = all_outputs[layer.input_layer]
            except KeyError:
                # one of the input_layer attributes must have been `None`
                raise ValueError("get_output() was called without giving an "
                                 "input expression for the free-floating "
                                 "layer %r. Please call it with a dictionary "
                                 "mapping this layer to an input expression."
                                 % layer)
            all_outputs[layer] = layer.get_output_for(layer_inputs, **kwargs)
    # return the output(s) of the requested layer(s) only
    try:
        return [all_outputs[layer] for layer in layer_or_layers]
    except TypeError:
        return all_outputs[layer_or_layers]


def get_output_shape(layer_or_layers, input_shapes=None):
    """
    Computes the output shape of the network at one or more given layers.

    :parameters:
        - layer_or_layers : Layer or list
            the :class:`Layer` instance for which to compute the output
            shapes, or a list of :class:`Layer` instances.

        - input_shapes : None, tuple, or dict
            If None, uses the input shapes associated with the
            :class:`InputLayer` instances.
            If a tuple, this defines the input shape for a single
            :class:`InputLayer` instance. Will throw a ValueError if there
            are multiple :class:`InputLayer` instances.
            If a dictionary, any :class:`Layer` instance (including the
            input layers) can be mapped to a shape tuple to use instead of
            its regular output shape.

    :returns:
        - output : tuple or list
            the output shape of the given layer(s) for the given network input
    """
    # shortcut: return precomputed shapes if we do not need to propagate any
    if input_shapes is None or input_shapes == {}:
        try:
            return [layer.output_shape for layer in layer_or_layers]
        except TypeError:
            return layer_or_layers.output_shape

    from .input import InputLayer
    from .base import MergeLayer
    # obtain topological ordering of all layers the output layer(s) depend on
    if isinstance(input_shapes, dict):
        treat_as_input = input_shapes.keys()
    else:
        treat_as_input = []

    all_layers = get_all_layers(layer_or_layers, treat_as_input)
    # initialize layer-to-shape mapping from all input layers
    all_shapes = dict((layer, layer.shape)
                      for layer in all_layers
                      if isinstance(layer, InputLayer) and
                      layer not in treat_as_input)
    # update layer-to-shape mapping from given input(s), if any
    if isinstance(input_shapes, dict):
        all_shapes.update(input_shapes)
    elif input_shapes is not None:
        if len(all_shapes) > 1:
            raise ValueError("get_output_shape() was called with a single "
                             "input shape on a network with multiple input "
                             "layers. Please call it with a dictionary of "
                             "input shapes instead.")
        for input_layer in all_shapes:
            all_shapes[input_layer] = input_shapes
    # update layer-to-shape mapping by propagating the input shapes
    for layer in all_layers:
        if layer not in all_shapes:
            try:
                if isinstance(layer, MergeLayer):
                    input_shapes = [all_shapes[input_layer]
                                    for input_layer in layer.input_layers]
                else:
                    input_shapes = all_shapes[layer.input_layer]
            except KeyError:
                raise ValueError("get_output() was called without giving an "
                                 "input shape for the free-floating layer %r. "
                                 "Please call it with a dictionary mapping "
                                 "this layer to an input shape."
                                 % layer)
            all_shapes[layer] = layer.get_output_shape_for(input_shapes)
    # return the output shape(s) of the requested layer(s) only
    try:
        return [all_shapes[layer] for layer in layer_or_layers]
    except TypeError:
        return all_shapes[layer_or_layers]


def get_all_params(layer):
    """
    This function gathers all learnable parameters of all layers below one
    or more given :class:`Layer` instances, including the layer(s) itself.
    Its main use is to collect all parameters of a network just given the
    output layer(s).

    :usage:
        >>> from lasagne.layers import InputLayer, DenseLayer
        >>> l_in = InputLayer((100, 20))
        >>> l1 = DenseLayer(l_in, num_units=50)
        >>> all_params = get_all_params(l1)
        >>> all_params == [l1.W, l1.b]
        True

    :parameters:
        - layer : Layer or list
            the :class:`Layer` instance for which to gather all parameters,
            or a list of :class:`Layer` instances.

    :returns:
        - params : list
            a list of Theano shared variables representing the parameters.
    """
    layers = get_all_layers(layer)
    params = sum([l.get_params() for l in layers], [])
    return utils.unique(params)


def get_all_bias_params(layer):
    """
    This function gathers all learnable bias parameters of all layers below one
    or more given :class:`Layer` instances, including the layer(s) itself.

    This is useful for situations where the biases should be treated
    separately from other parameters, e.g. they are typically excluded from
    L2 regularization.

    :usage:
        >>> from lasagne.layers import InputLayer, DenseLayer
        >>> l_in = InputLayer((100, 20))
        >>> l1 = DenseLayer(l_in, num_units=50)
        >>> all_params = get_all_bias_params(l1)
        >>> all_params == [l1.b]
        True

    :parameters:
        - layer : Layer or list
            the :class:`Layer` instance for which to gather all bias
            parameters, or a list of :class:`Layer` instances.

    :returns:
        - params : list
            a list of Theano shared variables representing the bias parameters.

    """
    layers = get_all_layers(layer)
    params = sum([l.get_bias_params() for l in layers], [])
    return utils.unique(params)


def get_all_non_bias_params(layer):
    """
    This function gathers all learnable non-bias parameters of all layers below
    one or more given :class:`Layer` instances, including the layer(s) itself.

    This is useful for situations where the biases should be treated
    separately from other parameters, e.g. they are typically excluded from
    L2 regularization.

    :usage:
        >>> from lasagne.layers import InputLayer, DenseLayer
        >>> l_in = InputLayer((100, 20))
        >>> l1 = DenseLayer(l_in, num_units=50)
        >>> all_params = get_all_non_bias_params(l1)
        >>> all_params == [l1.W]
        True

    :parameters:
        - layer : Layer or list
            the :class:`Layer` instance for which to gather all non-bias
            parameters, or a list of :class:`Layer` instances.

    :returns:
        - params : list
            a list of Theano shared variables representing the non-bias
            parameters.

    """
    all_params = get_all_params(layer)
    all_bias_params = get_all_bias_params(layer)
    return [p for p in all_params if p not in all_bias_params]


def count_params(layer):
    """
    This function counts all learnable parameters (i.e. the number of scalar
    values) of all layers below one or more given :class:`Layer` instances,
    including the layer(s) itself.

    This is useful to compare the capacity of various network architectures.
    All parameters returned by the :class:`Layer`s' `get_params` methods are
    counted, including biases.

    :usage:
        >>> from lasagne.layers import InputLayer, DenseLayer
        >>> l_in = InputLayer((100, 20))
        >>> l1 = DenseLayer(l_in, num_units=50)
        >>> param_count = count_params(l1)
        >>> param_count
        1050
        >>> param_count == 20 * 50 + 50  # 20 input * 50 units + 50 biases
        True

    :parameters:
        - layer : Layer or list
            the :class:`Layer` instance for which to count the parameters,
            or a list of :class:`Layer` instances.
    :returns:
        - count : int
            the total number of learnable parameters.

    """
    params = get_all_params(layer)
    shapes = [p.get_value().shape for p in params]
    counts = [np.prod(shape) for shape in shapes]
    return sum(counts)


def get_all_param_values(layer):
    """
    This function returns the values of the parameters of all layers below one
    or more given :class:`Layer` instances, including the layer(s) itself.

    This function can be used in conjunction with set_all_param_values to save
    and restore model parameters.

    :usage:
        >>> from lasagne.layers import InputLayer, DenseLayer
        >>> l_in = InputLayer((100, 20))
        >>> l1 = DenseLayer(l_in, num_units=50)
        >>> all_param_values = get_all_param_values(l1)
        >>> (all_param_values[0] == l1.W.get_value()).all()
        True
        >>> (all_param_values[1] == l1.b.get_value()).all()
        True

    :parameters:
        - layer : Layer or list
            the :class:`Layer` instance for which to gather all parameter
            values, or a list of :class:`Layer` instances.

    :returns:
        - param_values : list of numpy.array
            a list of numpy arrays representing the parameter values.
    """
    params = get_all_params(layer)
    return [p.get_value() for p in params]


def set_all_param_values(layer, values):
    """
    Given a list of numpy arrays, this function sets the parameters of all
    layers below one or more given :class:`Layer` instances (including the
    layer(s) itself) to the given values.

    This function can be used in conjunction with get_all_param_values to save
    and restore model parameters.

    :usage:
        >>> from lasagne.layers import InputLayer, DenseLayer
        >>> l_in = InputLayer((100, 20))
        >>> l1 = DenseLayer(l_in, num_units=50)
        >>> all_param_values = get_all_param_values(l1)
        >>> # all_param_values is now [l1.W.get_value(), l1.b.get_value()]
        >>> # ...
        >>> set_all_param_values(l1, all_param_values)
        >>> # the parameter values are restored.

    :parameters:
        - layer : Layer or list
            the :class:`Layer` instance for which to set all parameter
            values, or a list of :class:`Layer` instances.
        - values : list of numpy.array
            a list of numpy arrays representing the parameter values,
            must match the number of parameters
    """
    params = get_all_params(layer)
    if len(params) != len(values):
        raise ValueError("mismatch: got %d values to set %d parameters" %
                         (len(values), len(params)))
    for p, v in zip(params, values):
        p.set_value(v)
