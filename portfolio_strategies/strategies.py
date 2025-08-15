from .distributions import alpha_sharp, alpha_sharp_brute_force
from .main_strategy import Strategy
from functools import partial
from .distributions import *


class StrategyFactory:
    def __init__(self, base_color: str):
        self.base_color = base_color

    def create(self, name: str, strategy_func, parent_class=Strategy):
        def _init_(self_):
            parent_class.__init__(self_, strategy_func)

        return type(
            name,
            (parent_class,),
            {'__init__': _init_, 'color': self.base_color}
        )

alpha_markowitz_factory = StrategyFactory("#ff0000")
convex_markowitz_factory = StrategyFactory("#191970")
markowitz_factory = StrategyFactory("#1f77b4")
riskfolio_factory = StrategyFactory("#ff7f0e")
herc_factory = StrategyFactory("#7f7f7f")
hrp_factory = StrategyFactory("#e377c2")


ConvexMarkowitzMinRisk = convex_markowitz_factory.create("ConvexMarkowitzMinRisk", cvx_min_risk)
ConvexMarkowitzMaxRet  = convex_markowitz_factory.create("ConvexMarkowitzMaxRet", cvx_max_profit)
ConvexMarkowitzSharp   = convex_markowitz_factory.create("ConvexMarkowitzSharp", cvx_sharp)
ConvexMarkowitzSharpShort = convex_markowitz_factory.create("ConvexMarkowitzSharpShort", partial(cvx_sharp, short=True))

ConvexMarkowitzSharpAlpha = lambda a :alpha_markowitz_factory.create(f"ConvexMarkowitzSharpAlpha_new_{int(a * 100)}_{100 - int(a * 100)}_", partial(alpha_sharp, a=a))
ConvexMarkowitzSharpBruteForceRF = lambda rf :alpha_markowitz_factory.create(f"ConvexMarkowitzSharpAlphaBruteForce_{rf}_", partial(alpha_sharp_brute_force, risk_free_rate=rf))

MarkowitzMinRisk = markowitz_factory.create("MarkowitzMinRisk", scp_min_risk)
MarkowitzMaxRet  = markowitz_factory.create("MarkowitzMaxRet", scp_max_profit)
MarkowitzSharp   = markowitz_factory.create("MarkowitzSharp", scp_sharp)

RiskfolioMinRisk = riskfolio_factory.create("RiskfolioMinRisk", riskfolio_min_risk)
RiskfolioMaxRet  = riskfolio_factory.create("RiskfolioMaxRet", riskfolio_max_profit)
RiskfolioSharp   = riskfolio_factory.create("RiskfolioSharp", riskfolio_sharp)
RiskfolioSharpShort = riskfolio_factory.create("RiskfolioSharpShort", partial(riskfolio_sharp, short=True))

HERCSharp   = herc_factory.create("HERCSharp", herc_sharp)
HERCMaxRet  = herc_factory.create("HERCMaxRet", herc_max_profit)
HERCMinRisk = herc_factory.create("HERCMinRisk", herc_min_risk)

HRPSharp   = hrp_factory.create("HRPSharp", hrp_sharp)
HRPMaxRet  = hrp_factory.create("HRPMaxRet", hrp_max_profit)
HRPMinRisk = hrp_factory.create("HRPMinRisk", hrp_min_risk)


class StrategyBTC(Strategy):
    def __init__(self):
        super().__init__()
        self._fee_rate = 0.0
        self._weights = {'BTC': 1.0}
        self.color = "#bcbd22"


class PortfolioOldCoins(Strategy):
    def __init__(self):
        super().__init__()
        self._fee_rate = 0.0
        self.color = "#800000"
        self.coins = ['BNB', 'ETH', 'LTC', 'NEO']


class PortfolioNewCoins(Strategy):
    def __init__(self):
        super().__init__()
        self._fee_rate = 0.0
        self.color = "#006400"
        self.coins = [
            'BNB', 'ETH', 'LTC', 'NEO', 'ADA', 'XRP', 'QNT',
            'TRX', 'TSUD', 'XLM', 'QTUM', 'ICX', 'IOTA', 'ETC'
        ]


MIN_RISK = [
    RiskfolioMinRisk(),
    MarkowitzMinRisk(),
    HRPMinRisk(),
    HERCMinRisk(),
    ConvexMarkowitzMinRisk(),
]

SHARP = [
    MarkowitzSharp(),
    RiskfolioSharp(),
    HRPSharp(),
    HERCSharp(),
    ConvexMarkowitzSharp(),
]

SHARP_SHORT = [
    ConvexMarkowitzSharpShort(),
    RiskfolioSharpShort(),
]

MAX_RET = [
    RiskfolioMaxRet(),
    ConvexMarkowitzMaxRet(),
    MarkowitzMaxRet(),
    HRPMaxRet(),
    HERCMaxRet(),
]


PORTFOLIO = [
    PortfolioOldCoins(),
    PortfolioNewCoins()
]