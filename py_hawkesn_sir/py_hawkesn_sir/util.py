def int_pairs_leq_n_when_summed(n):
    """
    Construct a list of pairs of integers (>=0) such that the sum of each pair
    is less or equal `n`.

    Parameters
    ----------
    n : int
        Upper bound for the sum of each pair.

    Returns
    -------
    pairs_list : list
        A list of tuples. Each tuple consists of two integers, both of which
        are >= 0. Each tuple has the property that the sum of two entries is
        less or equal `n`.

    Examples
    --------
    >>> import numpy as np
    >>> obtained = int_pairs_leq_n_when_summed(3)
    >>> desired = [(0, 0),
    ...            (0, 1),
    ...            (0, 2),
    ...            (0, 3),
    ...            (1, 0),
    ...            (1, 1),
    ...            (1, 2),
    ...            (2, 0),
    ...            (2, 1),
    ...            (3, 0)]
    >>> np.array_equal(obtained, desired)
    True
    """
    return [(x, y) for x in range(n + 1) for y in range(n + 1 - x)]


def get_index(x, y, size):
    """
    Parameters
    ----------
    x : int
        An integer in :math:`\\{0, 1, \\ldots, size\\}` such that
        :math:`x + y \\leq size`
    y : int
        Same as parameter x.
    size : int
        Upper bound for the sum of x and y.

    Returns
    -------
    index : int
        The index of the pair (x, y) in the list constructed by
        :func:`int_pairs_leq_n_when_summed(size)`.

    Examples
    --------
    >>> size = 3
    >>> pairs = int_pairs_leq_n_when_summed(size)
    >>> xy = (2,1)
    >>> desired = pairs.index(xy)
    >>> obtained = get_index(*xy, size)
    >>> desired == obtained
    True
    >>> xy = (1,2)
    >>> desired = pairs.index(xy)
    >>> obtained = get_index(*xy, size)
    >>> desired == obtained
    True
    """
    return y + sum(range(size+1, size+1-x, -1))


def int_triples_leq_n_when_summed(n):
    """
    Construct a list of 3-tuples of integers (>=0) such that the sum of each
    tuple is less or equal `n`.

    Parameters
    ----------
    n : int
        Upper bound for the sum of each 3-tuple.

    Returns
    -------
    triples_list : list
        A list of 3-tuples. Each tuple consists of integers, all of which are
        >= 0. Each tuple has the property that the sum of all entries is less
        or equal `n`.

    Examples
    --------
    >>> import numpy as np
    >>> obtained = int_triples_leq_n_when_summed(3)
    >>> desired = [(0, 0, 0),
    ...            (0, 0, 1),
    ...            (0, 0, 2),
    ...            (0, 0, 3),
    ...            (0, 1, 0),
    ...            (0, 1, 1),
    ...            (0, 1, 2),
    ...            (0, 2, 0),
    ...            (0, 2, 1),
    ...            (0, 3, 0),
    ...            (1, 0, 0),
    ...            (1, 0, 1),
    ...            (1, 0, 2),
    ...            (1, 1, 0),
    ...            (1, 1, 1),
    ...            (1, 2, 0),
    ...            (2, 0, 0),
    ...            (2, 0, 1),
    ...            (2, 1, 0),
    ...            (3, 0, 0)]
    >>> np.array_equal(obtained, desired)
    True
    """
    return [(x, y, z)
            for x in range(n + 1)
            for y in range(n + 1 - x)
            for z in range(n + 1 - x - y)]


def get_index_for_triples(x, y, z, size):
    """
    Parameters
    ----------
    x : int
        An integer in :math:`\\{0, 1, \\ldots, size\\}` such that
        :math:`x + y \\leq size`
    y : int
        Same as parameter x.
    z : int
        Same as parameter x.
    size : int
        Upper bound for the sum of x, y, and z.

    Returns
    -------
    index : int
        The index of the 3-tuple (x, y, z) in the list constructed by
        :func:`int_triples_leq_n_when_summed(size)`.

    Examples
    --------
    >>> size = 3
    >>> triples = int_triples_leq_n_when_summed(size)
    >>> xyz = (2,1,0)
    >>> desired = triples.index(xyz)
    >>> obtained = get_index_for_triples(*xyz, size)
    >>> desired == obtained
    True
    >>> xyz = (1,1,1)
    >>> desired = triples.index(xyz)
    >>> obtained = get_index_for_triples(*xyz, size)
    >>> desired == obtained
    True
    """
    return sum(sum(range(size+2-i)) for i in range(x)) + \
        sum(range(size-x+1, size-x+1 - y, -1)) + \
        z

