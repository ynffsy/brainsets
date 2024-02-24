from kirby.taxonomy import REACHING

# TODO: fix this hack. It's not a good idea because we don't have disambiguation by task
# type.
weight_registry = {
    int(REACHING.RANDOM): 1.0,
    int(REACHING.HOLD): 0.1,
    int(REACHING.REACH): 5.0,
    int(REACHING.RETURN): 1.0,
    int(REACHING.INVALID): 0.1,
    int(REACHING.OUTLIER): 0.,
}
