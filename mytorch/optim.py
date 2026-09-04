"""Device-preserving optimizers."""

from .backend import device_of, get_array_module, to_device


class Optimizer:
    def __init__(self, params):
        self.params = list(params)

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for parameter in self.params:
            parameter.grad = None

    zero_grad = reset_grad

    @staticmethod
    def _gradient_data(parameter):
        if parameter.grad is None:
            return None
        gradient = parameter.grad.realize_cached_data()
        if device_of(gradient) != parameter.device:
            raise ValueError(
                f"optimizer device mismatch: parameter is on {parameter.device}, "
                f"gradient is on {device_of(gradient)}"
            )
        return gradient


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for parameter in self.params:
            gradient = self._gradient_data(parameter)
            if gradient is None:
                continue
            data = parameter.realize_cached_data()
            xp = get_array_module(data)
            gradient = gradient + self.weight_decay * data
            if self.momentum:
                velocity = self.u.get(parameter)
                if velocity is None:
                    velocity = xp.zeros_like(data)
                elif device_of(velocity) != parameter.device:
                    velocity = to_device(velocity, parameter.device, dtype=data.dtype)
                velocity = self.momentum * velocity + (1 - self.momentum) * gradient
                self.u[parameter] = velocity.astype(data.dtype, copy=False)
                gradient = velocity
            parameter.cached_data = (data - self.lr * gradient).astype(
                data.dtype, copy=False
            )

    def clip_grad_norm(self, max_norm=0.25):
        """Clip each device-local parameter set without copying arrays to host."""
        groups = {}
        for parameter in self.params:
            gradient = self._gradient_data(parameter)
            if gradient is not None:
                groups.setdefault(parameter.device, []).append((parameter, gradient))
        for values in groups.values():
            xp = get_array_module(values[0][1])
            total = xp.zeros((), dtype=values[0][1].dtype)
            for _, gradient in values:
                total = total + xp.sum(gradient * gradient)
            scale = xp.minimum(1, max_norm / (xp.sqrt(total) + 1e-12))
            for parameter, gradient in values:
                parameter.grad.cached_data = (gradient * scale).astype(
                    gradient.dtype, copy=False
                )


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        if lr <= 0:
            raise ValueError("Adam lr must be positive")
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("Adam beta1 and beta2 must be in [0, 1)")
        if eps <= 0:
            raise ValueError("Adam eps must be positive")
        if weight_decay < 0:
            raise ValueError("Adam weight_decay must be non-negative")
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        for parameter in self.params:
            gradient = self._gradient_data(parameter)
            if gradient is None:
                continue
            data = parameter.realize_cached_data()
            xp = get_array_module(data)
            gradient = gradient + self.weight_decay * data
            previous_m = self.m.get(parameter)
            previous_v = self.v.get(parameter)
            if previous_m is None:
                previous_m = xp.zeros_like(data)
                previous_v = xp.zeros_like(data)
            elif device_of(previous_m) != parameter.device:
                previous_m = to_device(previous_m, parameter.device, dtype=data.dtype)
                previous_v = to_device(previous_v, parameter.device, dtype=data.dtype)
            moment = self.beta1 * previous_m + (1 - self.beta1) * gradient
            variance = self.beta2 * previous_v + (1 - self.beta2) * gradient * gradient
            self.m[parameter] = moment.astype(data.dtype, copy=False)
            self.v[parameter] = variance.astype(data.dtype, copy=False)
            moment_hat = moment / (1 - self.beta1 ** self.t)
            variance_hat = variance / (1 - self.beta2 ** self.t)
            parameter.cached_data = (
                data - self.lr * moment_hat / (xp.sqrt(variance_hat) + self.eps)
            ).astype(data.dtype, copy=False)
