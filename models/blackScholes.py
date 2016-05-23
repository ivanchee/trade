from scipy.stats import norm
import math

# see www.econ-pol.unisi.it/fm10/greeksBS.pdf
class BlackScholes:
    # q is the continuous dividend rate
    def __init__(self, type, S, K, T, r, v, q):
        self.type = type
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.v = v
        self.q = q
        
        self.d1 = (math.log(self.S / self.K) + (self.r - self.q + 0.5 * math.pow(self.v, 2)) * self.T) / (self.v * math.sqrt(self.T))
        self.d2 = self.d1 - self.v * math.sqrt(self.T)

        self.Nd1 = norm.cdf(self.d1)
        self.Nd2 = norm.cdf(self.d2)
        self.NMd1 = norm.cdf(-self.d1)
        self.NMd2 = norm.cdf(-self.d2)

        self.NPd1 = norm.pdf(self.d1)
        self.NPd2 = norm.pdf(self.d2)
        self.NPMd1 = norm.pdf(-self.d1)
        self.NPMd2 = norm.pdf(-self.d2)

    def px(self):
        px = self.S * math.exp(-self.q * self.T) * self.Nd1 - self.K * math.exp(-self.r * self.T) * self.Nd2

        # put-call parity
        if 'P' == self.type:
            px = px + self.K * math.exp(-self.r * self.T) - self.S * math.exp(-self.q * self.T)

        return px

    def delta(self):
        if 'C' == self.type:
            return self.Nd1 * math.exp(-self.q * self.T)
        else:
            return (self.Nd1 - 1.0) * math.exp(-self.q * self.T)

    def gamma(self):
        return self.K * math.exp(-self.r * self.T) * self.NPd2 / (self.S * self.S * self.v * math.sqrt(self.T))

    # annualized rate of decay
    def theta(self):
        if 'C' == self.type:
            p1 = -1.0 * (self.S * self.v * math.exp(-self.q * self.T) / (2.0 * math.sqrt(self.T) * math.sqrt(2.0 * math.pi)) * math.exp(-1 * self.d1 * self.d1 / 2.0))
            p2 = self.r * self.K * math.exp(-self.r * self.T) * self.Nd2
            p3 = self.q * self.S * math.exp(-self.q * self.T) * self.Nd1
            return p1 - p2 + p3
        else:
            p1 = -1.0 * (self.S * self.v * math.exp(-self.q * self.T) / (2.0 * math.sqrt(self.T) * math.sqrt(2.0 * math.pi)) * math.exp(-1 * self.d1 * self.d1 / 2.0))
            p2 = self.r * self.K * math.exp(-self.r * self.T) * self.NMd2
            p3 = self.q * self.S * math.exp(-self.q * self.T) * self.NMd1
            return p1 + p2 - p3

    def rho(self):
        if 'C' == self.type:
            return self.T * self.K * math.exp(-self.r * self.T) * self.Nd2
        else:
            return self.T * self.K * math.exp(-self.r * self.T) * self.NMd2

    def vega(self):
        return math.sqrt(self.T) * self.K * math.exp(-self.r * self.T) * self.NPd2

    def all(self):
        d = {}
        d['type'] = self.type
        d['px'] = self.px()
        d['delta'] = self.delta()
        d['gamma'] = self.gamma()
        d['theta'] = self.theta()
        d['rho'] = self.rho()
        d['vega'] = self.vega()
        return d


if __name__ == "__main__":
    S = 100.0
    K = 102.0
    T = 1.0
    r = 0.05
    v = 0.20
    q = 0.02

    opt_call = BlackScholes('C', S, K, T, r, v, q)
    print('call : ' + str(opt_call.all()))

    opt_put = BlackScholes('P', S, K, T, r, v, q)
    print('put : ' + str(opt_put.all()))


