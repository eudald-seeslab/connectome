import logging
import time

logger = logging.getLogger("ct")


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time} seconds")
        return result

    return wrapper


def get_size_of_tensor(x):
    return (x.indices().nelement() * 8 + x.values().nelement() * 4) / 1024**2


def model_summary(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.debug(f"{name}: {param.shape}")
    logger.debug(
        f"The model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters."
    )


# Create logger
def get_logger(name, debug=False):
    log_level = logging.DEBUG if debug else logging.INFO
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.propagate = False
    return logger
