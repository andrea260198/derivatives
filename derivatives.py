import numpy as np
import copy
import math
import itertools


def Bond(r, S0, dates, c):
    dates = np.array(dates)
    delta_dates = np.append(np.array(0), dates)
    delta_dates = np.diff(delta_dates)
    B = np.sum(np.exp(-r * dates) * c * delta_dates * S0) + np.exp(-r * dates[-1]) * S0
    return B


def EuropeanPutOption(r, dt, S0, div, u, v, dates, alpha):
    T = dates[-1]  # define the last expiry date
    N = int(T / dt)  # number of steps
    S = [S0 * (u ** (N - k)) * (v ** k) for k in range(N + 1)]
    p = (1 - v - div * dt + r * dt) / (u - v)  # risk-neutral probability
    q = 1 - p
    K = alpha * S0  # strike

    V = [max(K - ST, 0) for ST in S]
    while len(V) > 1:
        V_up = V[0:-1]
        V_down = V[1:]
        V = [(p * V_up[k] + q * V_down[k]) / (1 + r * dt) for k in range(len(V_up))]

    return V[0]


def EuropeanCallOption(r, dt, S0, div, u, v, dates, alpha):
    K = alpha * S0
    # I use put-call parity
    V = EuropeanPutOption(r, dt, S0, div, u, v, dates, alpha) + S0 - K * math.exp(-r * dates[-1])
    return V


def NoBarrierNoCallable(r, dt, S0, div, u, v, dates, c, alpha):
    V = Bond(r, S0, dates, c) - EuropeanPutOption(r, dt, S0, div, u, v, dates, alpha)
    return V


def EuropeanPutOptionBarrierOut(r, dt, S0, div, u, v, dates, alpha, beta):
    T = dates[-1]  # define the last expiry date
    N = int(T / dt)  # number of steps
    S = [S0 * (u ** (N - k)) * (v ** (k)) for k in range(N + 1)]
    p = (1 - v - div * dt + r * dt) / (u - v)
    q = 1 - p
    K = alpha * S0  # strike

    V = []
    for ST in S:
        if ST > beta * S0:
            V = V + [max(K - ST, 0)]
        else:
            V = V + [0]
    while len(V) > 1:
        V_up = V[0:-1]
        V_down = V[1:]
        V = [(p * V_up[k] + q * V_down[k]) / (1 + r * dt) for k in range(len(V_up))]
        S = [S0 * (u ** (N - k)) * (v ** (k)) for k in range(len(V_up))]
        for ST in S:
            k = 0
            if ST <= beta * S0:
                V[k] = 0
            k = k + 1

    return V[0]


def EuropeanPutOptionBarrierIn(r, dt, S0, div, u, v, dates, alpha, beta):
    V = EuropeanPutOption(r, dt, S0, div, u, v, dates, alpha) - EuropeanPutOptionBarrierOut(r, dt, S0, div, u, v, dates,
                                                                                            alpha, beta)
    return V


def BarrierNoCallable(r, dt, S0, div, u, v, dates, c, alpha, beta):
    V = Bond(r, S0, dates, c) - EuropeanPutOptionBarrierIn(r, dt, S0, div, u, v, dates, alpha, beta)
    return V


def NoBarrierCallable(r, dt, S0, div, u, v, dates, c, alpha):
    dates = np.array(dates)

    q = (1 - v - div * dt + r * dt) / (u - v)  # risk-free probability
    T = dates[-1]  # Expiry
    delta_dates = np.append(np.array(0), dates)
    delta_dates = np.diff(delta_dates)
    N = int(T / dt)  # Number of steps

    price_put = np.zeros((N + 1, N + 1))
    for j in range(N + 1):
        price_put[N, j] = 1 + c * delta_dates[-1] - max(0, alpha - (v ** (N - j)) * (
                u ** j))  # Value at expiry is principal + coupon - payoff

    for i in range(N - 1, -1, -1):

        for j in range(i + 1):
            price_put[i, j] = 1 / (1 + r * dt) * (q * price_put[i + 1, j + 1] + (1 - q) * price_put[i + 1, j])

            if i / (
                    1 / dt) in dates:  # here we need to do like this otherwise i*dt isn't correct due to approximation errors
                diff = delta_dates[dates == i / (1 / dt)]

                price_put[i, j] = min(1 + c * diff, price_put[i, j] + c * diff)

    return S0 * price_put[0, 0]


def BarrierCallable(r, dt, S0, div, u, v, dates, c, alpha, beta):
    dates = np.array(dates)
    T = dates[-1]  # define the last expiry date
    N = int(T / dt)  # number of steps
    delta_dates = np.append(np.array(0), dates)
    delta_dates = np.diff(delta_dates)
    q = (1 - v - div * dt + r * dt) / (u - v)

    V = np.zeros((1, 2 ** N))
    S = S0 * np.ones((1, 2 ** N))

    barrier = np.full((1, 2 ** N), False)

    for m in range(N):
        temp = S.copy()
        temp_bar = barrier.copy()
        for i in range(2 ** m):

            S[0, 2 * i + 1] = temp[0, i] * u

            S[0, 2 * i] = temp[0, i] * v

            if temp_bar[0, i] == True:
                barrier[0, 2 * i] = True
                barrier[0, 2 * i + 1] = True
            if S[0, 2 * i] / S0 <= beta:
                barrier[0, 2 * i] = True
            if S[0, 2 * i + 1] / S0 <= beta:
                barrier[0, 2 * i + 1] = True

    # Payoff at maturity
    for i in range(2 ** (N)):
        if barrier[0, i] == True:
            V[0, i] = -max(0, alpha - S[0, i] / S0) + 1 + c * delta_dates[-1]
        else:
            V[0, i] = 1 + c * delta_dates[-1]

    for m in range(N - 1, -1, -1):
        V_temp = V.copy()
        for i in range(2 ** m):
            V[0, i] = 1 / (1 + r * dt) * (q * V_temp[0, 2 * i + 1] + (1 - q) * V_temp[0, 2 * i])

            if m / (
                    1 / dt) in dates:  # here we need to do like this otherwise m*dt isn't correct due to approximation errors
                diff = delta_dates[dates == m / (1 / dt)]
                V[0, i] = min(1 + c * diff, V[0, i] + c * diff)

    return V[0, 0] * S0


def Delta(fun, **kwargs):
    """
    Example:
        Delta(NoBarrierCallable, r=0.01, dt=0.01, S0=100, div=0.02, u=1.01, v=0.99, dates=[1, 2, 3], c=0.01, alpha=1)
    """

    kwargs_up = copy.deepcopy(kwargs)
    kwargs_down = copy.deepcopy(kwargs)
    U = kwargs['u']
    D = kwargs['v']
    kwargs_up['S0'] = kwargs['S0'] * U
    kwargs_up['dates'] = list(np.array(kwargs['dates']) - kwargs['dt'])
    kwargs_down['S0'] = kwargs['S0'] * D
    kwargs_down['dates'] = list(np.array(kwargs['dates']) - kwargs['dt'])

    delta = (fun(**kwargs_up) - fun(**kwargs_down)) / ((U - D) * kwargs['S0'])
    return delta


def replicate(fun, **kwargs):
    V = fun(**kwargs)
    S0 = kwargs['S0']
    riskFreeAsset = V - Delta(fun, **kwargs) * S0
    riskyAsset = V - riskFreeAsset
    return riskFreeAsset, riskyAsset


def rDivCalibrate(data, S0):
    minPenalty = np.inf
    rRange = np.arange(-0.01, 0.05, 0.001)
    divRange = np.arange(0, 0.05, 0.001)
    print('Loading...')
    for r, div in itertools.product(rRange, divRange):
        penalty = 0

        for i in data.index:
            K = data.iloc[i]['Strike']
            C = data.iloc[i]['Call']
            P = data.iloc[i]['Put']

            penalty += (C - P - S0 * math.exp(-div) + K * math.exp(-r)) ** 2

        if penalty < minPenalty:
            minPenalty = penalty;
            r_cal, div_cal = r, div

        print("/", end="")

    return r_cal, div_cal


def uvCalibrate(data, r, dt, S0, div, dates):
    minPenalty = np.inf
    uRange = np.arange(1.00001, 1.5, 0.001)
    print('Loading...')
    for u in uRange:
        penalty = 0
        v = 2 - u + 2 * (r - div) * dt
        for i in data.index:
            K = data.iloc[i]['Strike']
            C = data.iloc[i]['Call']
            P = data.iloc[i]['Put']
            alpha = K / S0

            penalty += (EuropeanCallOption(r, dt, S0, div, u, v, dates, alpha) - C) ** 2
            penalty += (EuropeanPutOption(r, dt, S0, div, u, v, dates, alpha) - P) ** 2

        if penalty < minPenalty:
            minPenalty = penalty
            u_cal = u
            v_cal = v

        print("/", end="")

    return u_cal, v_cal


def cSolve(fun, **kwargs):
    cRange = np.arange(-0.02, 0.20, 0.01)
    minPenalty = np.inf
    S0 = kwargs['S0']
    for c in cRange:
        penalty = (fun(**kwargs, c=c) - S0) ** 2
        if penalty < minPenalty:
            minPenalty = penalty
            c_sol = c
    return c_sol

def alphaSolve(fun, **kwargs):
    alphaRange = np.arange(0.5, 2.3, 0.01)
    minPenalty = np.inf
    S0 = kwargs['S0']
    for alpha in alphaRange:
        penalty = (fun(**kwargs, alpha=alpha) - S0) ** 2
        if penalty < minPenalty:
            minPenalty = penalty
            alpha_sol = alpha
    return alpha_sol


def alphaSolveBarrier(fun, r, dt, S0, div, u, v, dates, c, betaCoeff):
    alphaRange = np.arange(0.5, 2.3, 0.01)
    minPenalty = np.inf
    for alpha in alphaRange:
        penalty = (fun(r=r, dt=dt, S0=S0, div=div, u=u, v=v, dates=dates, c=c, alpha=alpha, beta=betaCoeff*alpha) - S0) ** 2
        if penalty < minPenalty:
            minPenalty = penalty
            alpha_sol = alpha
    return alpha_sol