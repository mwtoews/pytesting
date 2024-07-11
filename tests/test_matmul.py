import numpy as np

print(np.__version__)
coords = np.frombuffer(
    b"\xcb\x92\x87\x89\xfe\xde4@\x92\xce\xf0W\\L,\xc0\xb3\x03b\xffN!3\xc0\xe1y"
    b"\x97b\x82\xb6+\xc0&ww\x99E\xe52\xc0\xfa\x8e\xbaY\xf7*0@\x04d\x85\xa0|"
    b"\x1a5@\x14\x14h\xaap\xac/@\xcb\x92\x87\x89\xfe\xde4@\x92\xce\xf0W\\L,\xc0"
).reshape(5, 2)
A = np.frombuffer(
    b"7\xb3\xf1\xdd\xc7\xff\xef\xbf\xeb\xe4b\x9b\xf5\xf7}?"
    b"\xeb\xe4b\x9b\xf5\xf7}\xbf7\xb3\xf1\xdd\xc7\xff\xef\xbf"
).reshape(2, 2)
print("A")
print(A)
print("coords")
print(coords)
print(coords[0, :] == coords[-1, :])
expected = np.matmul(A, coords.T).T


def check(result):
    print("expected")
    print(expected)
    print("result")
    print(result)
    assert result.shape == (5, 2)
    print(result[0, :] == result[-1, :])
    np.testing.assert_array_equal(result[0, :], result[-1, :])
    np.testing.assert_array_almost_equal_nulp(result, expected)


def test_matmul_1():
    result = np.matmul(A, coords.T).T
    check(result)


def test_matmul_2():
    result = np.matmul(coords, A.T)
    check(result)


def test_matmul_op_1():
    result = (A @ coords.T).T
    check(result)


def test_matmul_op_2():
    result = coords @ A.T
    check(result)


def test_dot_1():
    result = np.dot(A, coords.T).T
    check(result)


def test_dot_2():
    result = np.dot(coords, A.T)
    check(result)


def test_robust_matmul():
    x, y = coords.T
    a, b, d, e = A.ravel()
    xp = a * x + b * y
    yp = d * x + e * y
    result = np.stack([xp, yp]).T
    check(result)
