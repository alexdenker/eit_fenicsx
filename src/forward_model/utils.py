
import numpy as np 


def current_method(L, l, method=1, value=1):
    """
    Create a numpy array (or a list of arrays) that represents the current pattern in the electrodes.

    Taken from: https://github.com/HafemannE/FEIT_CBM34/blob/main/CBM/FEIT_codes/FEIT_onefile.py


    :param L: Number of electrodes.
    :type L: int
    :param l: Number of measurements.
    :type l: int
    :param method: Current pattern. Possible values are 1, 2, 3, or 4 (default=1).
    :type method: int
    :param value: Current density value (default=1).
    :type value: int or float

    :returns: list of arrays or numpy array -- Return list with current density in each electrode for each measurement.

    :Method Values:
        1. 1 and -1 in opposite electrodes.
        2. 1 and -1 in adjacent electrodes.
        3. 1 in one electrode and -1/(L-1) for the rest.
        4. For measurement k, we have: (sin(k*2*pi/16) sin(2*k*2*pi/16) ... sin(16*k*2*pi/16)).
        5. All against 1

    :Example:

    Create current pattern 1 with 4 measurements and 4 electrodes:

    >>> I_all = current_method(L=4, l=4, method=1)
    >>> print(I_all)
        [array([ 1.,  0., -1.,  0.]),
        array([ 0.,  1.,  0., -1.]),
        array([-1.,  0.,  1.,  0.]),
        array([ 0., -1.,  0.,  1.])]

    Create current pattern 2 with 4 measurements and 4 electrodes:

    >>> I_all = current_method(L=4, l=4, method=2)
    >>> print(I_all)
        [array([ 1., -1.,  0.,  0.]),
        array([ 0.,  1., -1.,  0.]),
        array([0.,  0.,  1., -1.]),
        array([ 1.,  0.,  0., -1.])]

    """
    I_all = []
    # Type "(1,0,0,0,-1,0,0,0)"
    if method == 1:
        if L % 2 != 0:
            raise Exception("L must be odd.")

        for i in range(l):
            if i <= L / 2 - 1:
                I = np.zeros(L)
                I[i], I[i + int(L / 2)] = value, -value
                I_all.append(I)
            elif i == L / 2:
                print(
                    "This method only accept until L/2 currents, returning L/2 currents."
                )
    # Type "(1,-1,0,0...)"
    if method == 2:
        for i in range(l):
            if i != L - 1:
                I = np.zeros(L)
                I[i], I[i + 1] = value, -value
                I_all.append(I)
            else:
                I = np.zeros(L)
                I[0], I[i] = -value, value
                I_all.append(I)
    # Type "(1,-1/15, -1/15, ....)"
    if method == 3:
        for i in range(l):
            I = np.ones(L) * -value / (L - 1)
            I[i] = value
            I_all.append(I)
    # Type "(sin(k*2*pi/16) sin(2*k*2*pi/16) ... sin(16*k*2*pi/16))"
    if method == 4:
        for i in range(l):
            I = np.ones(L)
            for k in range(L):
                I[k] = I[k] * np.sin((i + 1) * (k + 1) * 2 * np.pi / L)
            I_all.append(I)

    if method == 5:
        for i in range(l):
            if i <= L - 1:
                I = np.zeros(L)
                I[0] = -value
                I[i + 1] = value
                I_all.append(I)
            else:
                print(
                    "This method only accept until L-1 currents, returning L-1 currents."
                )

    if l == 1:
        I_all = I_all[0]
    return np.array(I_all)


if __name__ == "__main__":
    Inj_ref = current_method(L=16, l=16, method=2)

    print(Inj_ref)
    print(Inj_ref.shape)
