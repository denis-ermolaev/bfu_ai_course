def simple_probability(m, n):
    return m / n

def logical_or(m, k, n):
    return (m + k) / n

def logical_and(m, k, n, l):
    p_a = m / n
    p_b = k / l
    return p_a * p_b


def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))


def conditional_probability(values):
    count_first_is_1 = 0
    count_both_are_1 = 0

    for first, second in values:
        if first == 1:
            count_first_is_1 += 1
            if second == 1:
                count_both_are_1 += 1

    if count_first_is_1 == 0:
        return 0
    return count_both_are_1 / count_first_is_1


def bayesian_probability(a, b, ba):
    return (ba * a) / b