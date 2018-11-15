import numpy


class FactoryTaxProblem:
    c = [.8, 1, .95, .72, .9, .87, .72, .82, .93, .67]
    A_u = numpy.eye(10)
    A_u.append([1, 1] + [0] * 8)

    b_u = [
        1000, 1000, 3000, 1500, 1600, 750, 2000, 1100, 500, 350
    ]

    A_eq = numpy.zeros((10, 10))
    A_eq[0] = [0]*5 + [1, 1, 1, 0, 0]

    b_eq = [1800] + [0] * 9