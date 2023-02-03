import math

import numpy as np
import scipy.io as sio


lambdaa = 1  # radius of l2 ball
maxIter = 10000  # maximal number of iterations
tol = 1e-4  # the relative gap
debug = False  # the flag whether it is for debugging
delta = 1e-6


def epne(a0: np.ndarray, q0: np.ndarray):
    m = len(a0)
    n = len(q0)
    vga, vla, a = np.zeros([m], dtype=np.float64), np.zeros([m], dtype=np.float64), np.zeros([m, 1], dtype=np.float64)
    vgq, vlq, q = np.zeros([n], dtype=np.float64), np.zeros([n], dtype=np.float64), np.zeros([n, 1], dtype=np.float64)
    a0 = a0.astype(dtype=np.float64)
    q0 = q0.astype(dtype=np.float64)
    nua, nuq = m, n
    i, j, k, l = 0, 0, 0, 0
    vua, vuq = a0, q0
    sa, sq, sq0 = .0, .0, .0
    # compute the sum of q0
    i = 0
    while i < n:
        q0[i] = -q0[i]
        sq0 += q0[i]
        i += 1
    while nua + nuq > 0:
        if nua > 0:
            rho = vua[0]
        else:
            rho = vuq[0]
        dsa = 0.0
        dsq = 0.0
        nga = 0
        nla = 0
        nea = 0
        ngq = 0
        nlq = 0
        neq = 0

        i = 0
        while i < nua:  # split the vector a
            val = vua[i]
            if val > rho:
                vga[nga] = val
                nga += 1
                dsa += val
            elif val < rho:
                vla[nla] = val
                nla += 1
            else:
                dsa += val
                nea += 1
            i += 1

        i = 0
        while i < nuq:  # split the vector q
            val = vuq[i]
            if val > rho:
                vgq[ngq] = val
                ngq += 1
                dsq += val
            elif val < rho:
                vlq[nlq] = val
                nlq += 1
            i += 1

        df = sa + dsa + (sq0 - sq - dsq) - (k + nga + nea) * rho - (n - l - ngq) * rho
        if df < 0:
            vua = vla
            nua = nla
            sa += dsa
            k += (nga + nea)
            vuq = vlq
            nuq = nlq
            sq += dsq
            l += ngq
        else:
            vua = vga
            nua = nga
            vuq = vgq
            nuq = ngq

    rho = (sa + sq0 - sq) / (k + n - l)

    i = 0
    while i < m:
        val = a0[i] - rho
        if val > 0:
            a[i] = val
        else:
            a[i] = 0.0
        i += 1

    i = 0
    while i < n:
        q0[i] = -q0[i]
        val = q0[i] + rho
        if val > 0:
            q[i] = val
        else:
            q[i] = 0.0
        i += 1
    return a, q


def topPush(X: np.ndarray, y: np.ndarray):
    """
    :param X: X is a matrix, each row is an instance [n_instance, n_feature]
    :param y: y is the labels (+1 / -1) [n_instance, 1]
    :return: the learnt linear ranking model, [n_feature, 1]
    """
    ### initialization
    Xpos = X[y.squeeze(axis=1) == 1, :]  # positive instances
    Xneg = X[y.squeeze(axis=1) == -1, :]  # negative instances
    m = np.sum(y == 1)  # number of positive instances
    n = np.sum(y == -1)  # number of negative instances
    L = 1 / m
    a, ap, aap = np.zeros([m, 1]), np.zeros([m, 1]), np.zeros([m, 1])
    q, qp, qqp = np.zeros([n, 1]), np.zeros([n, 1]), np.zeros([n, 1])
    t = 1
    tp = 0
    stop_flag = False

    fun_val = np.zeros([maxIter, 1])

    ### Nesterov's method (To slove a and q)
    for iter in range(0, maxIter, 1):
        ## step1
        # compute search point s based on ap (qp) and a (q) (with beta)
        beta = (tp - 1) / t
        sa = a + beta * aap
        sq = q + beta * qqp

        ## step2
        # line search for L and compute the new approximate solution x

        v = np.dot(np.transpose(sa), Xpos) - np.dot(np.transpose(sq), Xneg)  # sa(m, 1), Xpos(m , len), v(1, len)
        # compute the gradient and function value at s
        gd_a = np.dot(Xpos, np.transpose(v)) / (
                    lambdaa * m) + sa / 2 - 1  # Xpos(m, len), v(1, len), sa(m, 1) gradient on a
        gd_q = np.dot(Xneg, np.transpose(v)) / (-lambdaa * m)  # Xneg(n, len), v(1, len), gradient on q
        f_s = np.vdot(v, v) / (2 * lambdaa * m) - np.sum(sa) + np.vdot(sa, sa) / 4  # v(1, len) function value
        # set ap=a  qp=q
        ap = a
        qp = q
        while True:
            # let sa walk in a step in the anti-gradient of sa to get va and project va onto the line
            va = sa - gd_a / L  # sa(m, 1), gd_a(m, 1), va(m, 1)
            vq = sq - gd_q / L  # sq(n, 1), gd_q(n, 1), vq(n, 1)
            # euclidean projection onto the line (equality constraint)
            # [a, q, k] = proj_line(va, vq);
            a, q = epne(va, vq)
            # compute the objective value at the new approximate solution
            v = np.dot(np.transpose(a), Xpos) - np.dot(np.transpose(q),
                                                       Xneg)  # a(m, 1), Xpos(m, len), q(n, 1), Xneg(n, len), v(1, len)
            f_new = np.vdot(v, v) / (2 * lambdaa * m) - np.sum(a) + np.vdot(a, a) / 4  # v(len, ), a(m, 1)
            df_a = a - sa  # df_a(m, 1)
            df_q = q - sq  # df_q(n, 1)
            r_sum = np.vdot(df_a, df_a) + np.vdot(df_q, df_q)

            if math.sqrt(r_sum) <= delta:
                if debug:
                    print("\n The distance between the searching point and the approximate is {}, "
                          "and is smaller than {} \n".format(math.sqrt(r_sum, delta)))
                stop_flag = True
                break

            l_sum = f_new - f_s - np.vdot(gd_a, df_a) - np.vdot(gd_q, df_q)

            # the condition is l_sum <= L * r_sum
            if l_sum <= r_sum * L * 0.5:
                break
            else:
                L *= 2

        ## step3
        # update a and q, and check whether converge
        tp, t = t, (1 + math.sqrt(4 * t * t + 1)) / 2
        aap, qqp = a - ap, q - qp
        fun_val[iter] = f_new / m

        # check the stop condition
        if iter >= 10 and abs(fun_val[iter] - fun_val[iter - 1]) <= abs(fun_val[iter - 1]) * tol:
            if debug:
                print("\n Relative obj. gap is less than {} \n".format(tol))
            stop_flag = 1

        if stop_flag:
            break

        if debug:
            print("'{} : {}  {}\n".format(iter, fun_val[iter], L))
    return np.transpose(v) / (lambdaa * m)
