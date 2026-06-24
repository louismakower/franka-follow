def _add_agent(args_cli):
    if args_cli.agent.lower() == "ppo":
        args_cli.agent = "louis_rl_ppo_cfg_entry_point"
    elif args_cli.agent.lower() == "sac":
        args_cli.agent = "louis_rl_sac_cfg_entry_point"
    else:
        raise ValueError(f"Agent ({args_cli.agent}) not supported. Supported: ['ppo', 'sac']")
    return args_cli