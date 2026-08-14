"""LeRobot episode 关节角加载,含缺失右臂的 action 移位补齐。

数据约定:observation.state 14 维 = [左臂 6 关节, 左夹爪, 右臂 6 关节, 右夹爪]。
早期 episode 右臂 state 全零(未录制),按用户规则从 action 补齐:
state[0] = 0,state[t] = action[t-1]。
"""

import numpy as np
import pyarrow.parquet as pq


def load_episode_joints(parquet_path):
    """加载一集关节角。

    返回 dict:left_q (N,6)、left_grip (N,)、right_q (N,6)、right_grip (N,)、
    reconstructed: 被 action 补齐的臂名列表。
    """
    t = pq.read_table(parquet_path, columns=["action", "observation.state"])
    state = np.stack(t.column("observation.state").to_pylist()).astype(np.float64)
    action = np.stack(t.column("action").to_pylist()).astype(np.float64)

    reconstructed = []
    for side, sl in (("left", slice(0, 7)), ("right", slice(7, 14))):
        joints = state[:, sl]
        if np.all(joints == 0.0):
            state[1:, sl] = action[:-1, sl]  # state[t] = action[t-1],state[0] 保持 0
            reconstructed.append(side)

    return {
        "left_q": state[:, 0:6],
        "left_grip": state[:, 6],
        "right_q": state[:, 7:13],
        "right_grip": state[:, 13],
        "reconstructed": reconstructed,
    }
