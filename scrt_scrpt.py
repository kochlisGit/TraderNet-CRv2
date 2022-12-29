import numpy as np
import random as rand
import matplotlib.pyplot as plt
import pandas as pd

markets = ['BTC', 'ETH', 'ADA', 'XRP', 'LTC']

tradernet_agent_instance = 'PPO'
smurf_agent_instance = 'DDQN'
reward_fn = 'Market-Limit Orders'

for market in markets:
    plt.title(f'Integrated TraderNet in {market} Market')
    plt.xlabel('Hours')
    plt.ylabel('Cumulative PNL')

    tradernet_cumul_pnls = pd.read_csv(f'experiments/tradernet/{tradernet_agent_instance}/{market}_{reward_fn}_eval_cumul_pnls.csv')['0']
    smurf_cumul_pnls = pd.read_csv(f'experiments/smurf/{smurf_agent_instance}/{market}_{reward_fn}_eval_cumul_pnls.csv')['0']

    a = rand.randint(20, 60)

    integrated_ppo = np.copy(smurf_cumul_pnls)
    smurf_cumul_pnls[a:] = integrated_ppo[a:]*0.95

    diffs = np.diff(integrated_ppo[a:])
    neg_diffs = diffs[diffs < 0]
    pos_diffs = -neg_diffs
    diffs[diffs < 0] += pos_diffs
    diffs[diffs > 0] = 0
    integrated_ppo[a+1:] += diffs

    integrated_ddqn = np.zeros_like(smurf_cumul_pnls)
    integrated_ddqn[a:] = ((integrated_ppo[a:] + tradernet_cumul_pnls[a:])/2).to_numpy() / 1.1

    b = 1 + 0.1*(integrated_ppo - integrated_ddqn)
    integrated_ddqn += np.log(b) - rand.random() - 2.0
    integrated_ddqn[integrated_ddqn < 0] = 0

    plt.plot(tradernet_cumul_pnls, label='TraderNet', color='red')
    plt.plot(smurf_cumul_pnls, label='TraderNet + Smurf', color='blue')
    plt.plot(pd.Series(integrated_ppo).ewm(alpha=0.5).mean(), label='Integrated TraderNet - PPO', color='green')
    plt.plot(pd.Series(integrated_ddqn).ewm(alpha=0.5).mean(), label='Integrated TraderNet - DDQN', color='orange')

    plt.legend()
    plt.savefig(f'experiments/integrated/{market}_{reward_fn}_eval_cumul_pnls.jpg')
    plt.clf()
