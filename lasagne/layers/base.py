from collections import OrderedDict

from .. import utils


__all__ = [
    "Layer",
    "MultipleInputsLayer",
]


# Layer base class

class Layer(object):
    """
    The :class:`Layer` class represents a single layer of a neural network.
    It should be subclassed when implementing new types of layers.

    Because each layer can keep track of the layer(s) feeding into it, a
    network's output :class:`Layer` instance can double as a handle to the
    full network.
    """
    def __init__(self, incoming, name=None):
        """
        Instantiates the layer.

        :parameters:
            - incoming : a :class:`Layer` instance or a tuple
                the layer feeding into this layer, or the expected input shape
            - name : a string or None
                an optional name to attach to this layer
        """
        if isinstance(incoming, tuple):
            self.input_shape = incoming
            self.input_layer = None
        else:
            self.input_shape = incoming.output_shape
            self.input_layer = incoming

        self.name = name
        self.params = OrderedDict()

    @property
    def output_shape(self):
        return self.get_output_shape_for(self.input_shape)

    def get_params(self, **tags):
        """
        TODO: docstring
        """
        result = self.params.keys()

        only = set(tag for tag, value in tags.items() if value)
        if only:
            # retain all parameters that have all of the tags in `only`
            result = [param for param in result
                      if not (only - self.params[param])]

        exclude = set(tag for tag, value in tags.items() if not value)
        if exclude:
            # retain all parameters that have none of the tags in `exclude`
            result = [param for param in result
                      if not (self.params[param] & exclude)]

        return result

    def get_output_shape(self):
        """
        Deprecated. Use `layer.output_shape`.
        """
        import warnings
        warnings.warn("layer.get_output_shape() is deprecated and will be "
                      "removed for the first release of Lasagne. Please use "
                      "layer.output_shape instead.")
        return self.output_shape

    def get_output(self, input=None, **kwargs):
        """
        Deprecated. Use `lasagne.layers.get_output(layer, input, **kwargs)`.
        """
        import warnings
        warnings.warn("layer.get_output(...) is deprecated and will be "
                      "removed for the first release of Lasagne. Please use "
                      "lasagne.layers.get_output(layer, ...) instead.")
        from .helper import get_output
        return get_output(self, input, **kwargs)

    def get_output_shape_for(self, input_shape):
        """
        Computes the output shape of this layer, given an input shape.

        :parameters:
            - input_shape : tuple
                a tuple representing the shape of the input. The tuple should
                have as many elements as there are input dimensions, and the
                elements should be integers or `None`.

        :returns:
            - output : tuple
                a tuple representing the shape of the output of this layer.
                The tuple has as many elements as there are output dimensions,
                and the elements are all either integers or `None`.

        :note:
            This method will typically be overridden when implementing a new
            :class:`Layer` class. By default it simply returns the input
            shape. This means that a layer that does not modify the shape
            (e.g. because it applies an elementwise operation) does not need
            to override this method.
        """
        return input_shape

    def get_output_for(self, input, **kwargs):
        """
        Propagates the given input through this layer (and only this layer).

        :parameters:
            - input : Theano expression
                the expression to propagate through this layer

        :returns:
            - output : Theano expression
                the output of this layer given the input to this layer

        :note:
            This is called by the base :class:`Layer` implementation to
            propagate data through a network in `get_output()`. While
            `get_output()` asks the underlying layers for input and thus
            returns an expression for a layer's output in terms of the
            network's input, `get_output_for()` just performs a single step
            and returns an expression for a layer's output in terms of
            that layer's input.

            This method should be overridden when implementing a new
            :class:`Layer` class. By default it raises `NotImplementedError`.
        """
        raise NotImplementedError

    def add_param(self, spec, shape, name=None, **tags):
        # prefix the param name with the layer name if it exists
        if name is not None:
            if self.name is not None:
                name = "%s.%s" % (self.name, name)

        param = utils.create_param(spec, shape, name)
        # parameters should be trainable and regularizable by default
        tags['trainable'] = tags.get('trainable', True)
        tags['regularizable'] = tags.get('regularizable', True)
        self.params[param] = set(tag for tag, value in tags.items() if value)

        return param


class MultipleInputsLayer(Layer):
    """
    This class represents a layer that aggregates input from multiple layers.
    It should be subclassed when implementing new types of layers that
    obtain their input from multiple layers.
    """
    def __init__(self, incomings, name=None):
        """
        Instantiates the layer.

        :parameters:
            - incomings : a list of :class:`Layer` instances or tuples
                the layers feeding into this layer, or expected input shapes
            - name : a string or None
                an optional name to attach to this layer
        """
        self.input_shapes = [incoming if isinstance(incoming, tuple)
                             else incoming.output_shape
                             for incoming in incomings]
        self.input_layers = [None if isinstance(incoming, tuple)
                             else incoming
                             for incoming in incomings]
        self.name = name

    @Layer.output_shape.getter
    def output_shape(self):
        return self.get_output_shape_for(self.input_shapes)

    def get_output_shape_for(self, input_shapes):
        """
        Computes the output shape of this layer, given a list of input shapes.

        :parameters:
            - input_shape : list of tuple
                a list of tuples, with each tuple representing the shape of
                one of the inputs (in the correct order). These tuples should
                have as many elements as there are input dimensions, and the
                elements should be integers or `None`.

        :returns:
            - output : tuple
                a tuple representing the shape of the output of this layer.
                The tuple has as many elements as there are output dimensions,
                and the elements are all either integers or `None`.

        :note:
            This method must be overridden when implementing a new
            :class:`Layer` class with multiple inputs. By default it raises
            `NotImplementedError`.
        """
        raise NotImplementedError

    def get_output_for(self, inputs, **kwargs):
        """
        Propagates the given inputs through this layer (and only this layer).

        :parameters:
            - inputs : list of Theano expressions
                The Theano expressions to propagate through this layer

        :returns:
            - output : Theano expressions
                the output of this layer given the inputs to this layer

        :note:
            This is called by the base :class:`MultipleInputsLayer`
            implementation to propagate data through a network in
            `get_output()`. While `get_output()` asks the underlying layers
            for input and thus returns an expression for a layer's output in
            terms of the network's input, `get_output_for()` just performs a
            single step and returns an expression for a layer's output in
            terms of that layer's input.

            This method should be overridden when implementing a new
            :class:`Layer` class with multiple inputs. By default it raises
            `NotImplementedError`.
        """
        raise NotImplementedError
