import numpy as np

print(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).flatten())
flattened = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).flatten()
print(flattened.reshape([3, 3]))
tt = flattened.reshape([3, 3])
print(np.rot90(tt))
print(np.rot90(np.rot90(tt)))
print(np.rot90(np.rot90(np.rot90(tt))))
print(np.rot90(np.rot90(np.rot90(np.rot90(tt)))))
print(np.flip(tt, 0))
print(np.flip(np.flip(tt, 0), 1))
print(np.flip(np.flip(np.flip(tt, 0), 1), 0))

a = np.zeros([3, 3])
b = np.copy(a)
c = np.zeros([3, 3])

P = dict()
P[a.tobytes()] = b

print(b.tobytes() in P)
print(c.tobytes() in P)

a[0, 0] = 1
print(a.tobytes() in P)

a[0, 0] = 0
print(a.tobytes() in P)

print(np.frombuffer(b'\x00\x00\x00\x00\x00\x00\xf0\xbf\x00\x00\x00\x00\x00\x00\x00\x80\x00\x00\x00\x00\x00\x00\x00\x80\x00\x00\x00\x00\x00\x00\x00\x80\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00\x80\x00\x00\x00\x00\x00\x00\x00\x80\x00\x00\x00\x00\x00\x00\x00\x80\x00\x00\x00\x00\x00\x00\x00\x80'))

for key in P.keys():
    print(np.frombuffer(key))

print(type(np.array([])), np.array([]))