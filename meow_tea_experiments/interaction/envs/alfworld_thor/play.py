import yaml
from alfworld.agents.environment import get_environment # type: ignore

with open("config.yaml") as reader:
    config = yaml.safe_load(reader)

env_type = config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'
env = get_environment(env_type)(config, train_eval='train')
env = env.init_env(batch_size=1)
obs, info = env.reset()

while True:
    admissible_commands = list(info.get('admissible_commands', []))
    print("==========\n\n")
    print(admissible_commands)
    print("==========\n\n")

    try:
        action = input("Enter action (or 'quit' to exit): ").strip()
    except EOFError:
        break

    if not action:
        continue
    if action.lower() in {"quit", "exit"}:
        break

    obs, scores, dones, infos = env.step(action)
    print(f"Action: {action}, Obs: {obs[0]}")
    info = infos
    if hasattr(dones, "__iter__") and any(dones):
        print("Episode finished.")
        break
    if isinstance(dones, bool) and dones:
        print("Episode finished.")
        break