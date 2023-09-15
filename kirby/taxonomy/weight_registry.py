from kirby.tasks.reaching import REACHING

# TODO: fix this hack. It's not a good idea because we don't have disambiguation by task
# type.
weight_registry = {
    REACHING.RANDOM: 1.0,
    REACHING.HOLD: 1.0,
    REACHING.CENTER_OUT_REACH: 50.0,
    REACHING.CENTER_OUT_RETURN: 10.0,
    REACHING.INVALID: 1.0,
    REACHING.OUTLIER: 0.1,
    REACHING.CONTINUOUS_ACQUISITION: 1.0,
}